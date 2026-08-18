#!/usr/bin/env python
"""Safely build one case window of a BlendEMU pair-response catalogue.

This is the non-destructive counterpart of ``run_pipeline.py --steps 4`` for
response-only work.  It calls the production ``retrieve_response`` function,
streams case batches to one Feather/Arrow IPC file, and refuses to overwrite an
existing output.  Detection catalogues and other response products are not
touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from blendemu import response  # noqa: E402
from blendemu.config import load_config  # noqa: E402


def build_case(case, r_cfg, shear_cases, tile_name, data_path):
    frame = response.retrieve_response(
        case=case,
        r_max=r_cfg["r_max"],
        r_min=r_cfg.get("r_min", 0),
        k=r_cfg["k"],
        real="real0",
        tile_name=tile_name,
        data_path=data_path,
        shear_cases=shear_cases,
    )
    return frame[np.isfinite(frame["delta_et1"].to_numpy(float))].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-offset", required=True, type=int)
    parser.add_argument("--n-cases", required=True, type=int)
    parser.add_argument("--shear-cases", required=True, nargs=2)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--n-jobs", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
    )
    args = parser.parse_args()

    if args.case_offset < 0 or args.n_cases <= 0 or args.n_jobs <= 0:
        raise SystemExit("case-offset must be >=0; n-cases and n-jobs must be >0")
    output = os.path.abspath(args.output)
    if os.path.exists(output):
        raise SystemExit(f"refusing to overwrite existing {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    cfg = load_config(args.config)
    data_path = cfg["simulation"]["output_path"]
    tile_name = cfg["shape_measurement"]["tile_name"]
    r_cfg = cfg["catalogues"]["response"]
    start = args.case_offset
    stop = start + args.n_cases
    print(
        f"cases={start}..{stop - 1} shear={args.shear_cases[0]}->{args.shear_cases[1]} "
        f"r_max={r_cfg['r_max']} k={r_cfg['k']} n_jobs={args.n_jobs}\noutput={output}",
        flush=True,
    )

    writer = None
    schema = None
    n_rows = 0
    seen_cases = set()
    started = time.time()
    try:
        for chunk_start in range(start, stop, args.n_jobs):
            cases = list(range(chunk_start, min(chunk_start + args.n_jobs, stop)))
            frames = Parallel(n_jobs=len(cases))(
                delayed(build_case)(
                    case, r_cfg, args.shear_cases, tile_name, data_path,
                )
                for case in cases
            )
            if any(frame.empty for frame in frames):
                empty = [case for case, frame in zip(cases, frames) if frame.empty]
                raise RuntimeError(f"empty response frames for cases {empty}")
            part = pd.concat(frames, ignore_index=True)
            actual = set(part["case"].astype(int).unique())
            if actual != set(cases):
                raise RuntimeError(
                    f"case identity mismatch: expected {cases}, got {sorted(actual)}"
                )
            overlap = seen_cases & actual
            if overlap:
                raise RuntimeError(f"duplicate cases across chunks: {sorted(overlap)}")
            seen_cases.update(actual)
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = ipc.new_file(output, schema)
            else:
                table = table.cast(schema)
            for batch in table.to_batches(max_chunksize=65536):
                writer.write_batch(batch)
            n_rows += len(part)
            print(
                f"cases {cases[0]}..{cases[-1]}: +{len(part):,} rows "
                f"(total {n_rows:,}, {time.time() - started:.0f}s)",
                flush=True,
            )
    except Exception:
        if writer is not None:
            writer.close()
        if os.path.exists(output):
            failed = output + ".incomplete"
            if os.path.exists(failed):
                raise RuntimeError(f"both incomplete paths exist: {output}, {failed}")
            os.replace(output, failed)
            print(f"preserved partial output as {failed}", flush=True)
        raise
    else:
        if writer is None:
            raise RuntimeError("no response rows produced")
        writer.close()

    expected = set(range(start, stop))
    if seen_cases != expected:
        raise RuntimeError(f"missing cases: {sorted(expected - seen_cases)}")
    print(
        f"wrote {n_rows:,} rows over {len(seen_cases)} cases in "
        f"{time.time() - started:.0f}s\nRESPONSE_WINDOW_BUILD_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()

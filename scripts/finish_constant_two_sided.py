#!/usr/bin/env python
"""Finish missing products for an existing two-sided constant-shear run.

The simulation and per-case shape catalogues must already exist.  This script
does not touch either shear leg.  It streams a requested per-sign catalogue and
the paired response catalogue to new Arrow/Feather files, refusing to overwrite
any completed output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blendemu import response  # noqa: E402
from blendemu.config import load_config  # noqa: E402


SUMMARY_COLUMNS = [
    "case", "r_input_p", "Re_input_p", "response", "distance", "neighbored",
]
SELECTION_MAG_MIN = 18.0
SELECTION_MAG_MAX = 28.0
SELECTION_RE_MIN = 0.1
SELECTION_RE_MAX = 1.5
SELECTION_DISTANCE_MIN = 0.0
SELECTION_DISTANCE_MAX = 5.0


def _shape_path(data_path: str, case: int, shear: str, tile_name: str,
                shape_suffix: str) -> Path:
    return Path(data_path) / f"case{case}_{shear}" / "real0" / "catalogues" / \
        "Shapes" / f"shape_catalogue_detect_position{shape_suffix}_{tile_name}.feather"


def _check_inputs(data_path: str, cases: list[int], shear_labels: list[str],
                  tile_name: str, shape_suffix: str) -> None:
    missing = []
    for case in cases:
        for shear in shear_labels:
            path = _shape_path(data_path, case, shear, tile_name, shape_suffix)
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise RuntimeError(f"missing {len(missing)} per-case shape catalogues:\n{preview}{extra}")


def _stream_frames(output: str, cases: list[int], make_frame, finite_column: str | None,
                   label: str) -> int:
    output_path = Path(output).resolve()
    temporary = Path(str(output_path) + ".building")
    incomplete = Path(str(output_path) + ".incomplete")
    for path in (output_path, temporary, incomplete):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    schema = None
    total = 0
    seen: set[int] = set()
    started = time.time()
    try:
        for index, case in enumerate(cases, start=1):
            frame = make_frame(case)
            if finite_column is not None:
                finite = np.isfinite(frame[finite_column].to_numpy(float))
                frame = frame.loc[finite].reset_index(drop=True)
            if frame.empty:
                raise RuntimeError(f"empty {label} frame for case {case}")
            actual = set(frame["case"].astype(int).unique())
            if actual != {case}:
                raise RuntimeError(
                    f"{label} case identity mismatch: expected {case}, got {sorted(actual)}"
                )
            if case in seen:
                raise RuntimeError(f"duplicate {label} case {case}")
            seen.add(case)

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = ipc.new_file(str(temporary), schema)
            else:
                table = table.cast(schema)
            for batch in table.to_batches(max_chunksize=65536):
                writer.write_batch(batch)
            total += len(frame)
            print(
                f"{label}: case {case} ({index}/{len(cases)}), "
                f"+{len(frame):,} rows, total={total:,}, elapsed={time.time() - started:.0f}s",
                flush=True,
            )
    except Exception:
        if writer is not None:
            writer.close()
        if temporary.exists():
            os.replace(temporary, incomplete)
            print(f"preserved partial output as {incomplete}", flush=True)
        raise
    else:
        if writer is None:
            raise RuntimeError(f"no {label} rows produced")
        writer.close()
        if seen != set(cases):
            raise RuntimeError(f"missing {label} cases: {sorted(set(cases) - seen)}")
        os.replace(temporary, output_path)

    print(
        f"wrote {total:,} {label} rows over {len(seen)} cases to {output_path}",
        flush=True,
    )
    return total


def _accumulate_catalogue(path: str, case_min: int, case_max: int,
                          mag_max: float, re_min: float) -> dict:
    sums = {"all": defaultdict(float), "domain": defaultdict(float)}
    counts = {"all": defaultdict(int), "domain": defaultdict(int)}
    scanner = ds.dataset(path, format="feather").scanner(
        columns=SUMMARY_COLUMNS, batch_size=131072, use_threads=True,
    )
    for batch in scanner.to_batches():
        def values(name: str) -> np.ndarray:
            index = batch.schema.get_field_index(name)
            if index < 0:
                raise RuntimeError(f"{path}: missing required column {name}")
            return batch.column(index).to_numpy(zero_copy_only=False)

        case = values("case").astype(np.int64)
        mag = values("r_input_p").astype(float)
        size = values("Re_input_p").astype(float)
        value = values("response").astype(float)
        distance = values("distance").astype(float)
        neighbored = values("neighbored").astype(bool)
        in_window = (
            (case >= case_min) & (case <= case_max) & np.isfinite(value)
            & (mag > SELECTION_MAG_MIN) & (mag < SELECTION_MAG_MAX)
            & (size > SELECTION_RE_MIN) & (size < SELECTION_RE_MAX)
            & (
                ~neighbored
                | ((distance > SELECTION_DISTANCE_MIN) & (distance < SELECTION_DISTANCE_MAX))
            )
        )
        masks = {
            "all": in_window,
            "domain": in_window & (mag < mag_max) & (size > re_min),
        }
        for name, mask in masks.items():
            if not np.any(mask):
                continue
            local_case = case[mask]
            local_value = value[mask]
            for one_case in np.unique(local_case):
                selected = local_case == one_case
                sums[name][int(one_case)] += float(local_value[selected].sum())
                counts[name][int(one_case)] += int(selected.sum())

    expected = set(range(case_min, case_max + 1))
    result = {}
    for name in ("all", "domain"):
        present = {case for case, count in counts[name].items() if count > 0}
        if present != expected:
            raise RuntimeError(
                f"{path}: {name} case coverage mismatch; missing={sorted(expected - present)}, "
                f"extra={sorted(present - expected)}"
            )
        case_ids = sorted(expected)
        case_means = np.array(
            [sums[name][case] / counts[name][case] for case in case_ids], dtype=float,
        )
        n_rows = int(sum(counts[name].values()))
        total_sum = float(sum(sums[name].values()))
        case_sd = float(case_means.std(ddof=1))
        result[name] = {
            "n_rows": n_rows,
            "n_cases": len(case_ids),
            "row_weighted_mean": total_sum / n_rows,
            "case_weighted_mean": float(case_means.mean()),
            "case_sd": case_sd,
            "case_sem": case_sd / math.sqrt(len(case_ids)),
            "case_min": int(case_min),
            "case_max": int(case_max),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plus-shear", required=True)
    parser.add_argument("--minus-shear", required=True)
    parser.add_argument("--minus-output", required=True)
    parser.add_argument("--response-output", required=True)
    parser.add_argument("--reference-catalogue", required=True)
    parser.add_argument("--reference-case-min", required=True, type=int)
    parser.add_argument("--reference-case-max", required=True, type=int)
    parser.add_argument("--mag-max", type=float, default=25.8)
    parser.add_argument("--re-min", type=float, default=0.5)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_output).resolve()
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {summary_path}")
    if not Path(args.reference_catalogue).is_file():
        raise FileNotFoundError(args.reference_catalogue)

    cfg = load_config(args.config)
    sim_cfg = cfg["simulation"]
    cs_cfg = cfg["catalogues"]["constant_shear"]
    sm_cfg = cfg["shape_measurement"]
    data_path = sim_cfg["output_path"]
    start = int(sim_cfg.get("case_offset", 0))
    stop = start + int(sim_cfg["n_cases"])
    cases = list(range(start, stop))
    plus = str(args.plus_shear)
    minus = str(args.minus_shear)
    tile_name = sm_cfg["tile_name"]
    shape_suffix = cs_cfg.get("shape_suffix", "_all")
    include_isolated = bool(cs_cfg.get("include_isolated", True))
    r_max = cs_cfg["r_max"]
    r_min = cs_cfg.get("r_min", 0)
    k = cs_cfg["k"]

    expected_mag = abs(float(sim_cfg["constant_shear"]))
    if not np.isclose(float(plus), expected_mag) or not np.isclose(float(minus), -expected_mag):
        raise RuntimeError(
            f"requested shear labels {plus}, {minus} disagree with config magnitude {expected_mag}"
        )
    _check_inputs(data_path, cases, [plus, minus], tile_name, shape_suffix)

    print(
        f"finishing constant-shear cases {start}..{stop - 1}: {plus}/{minus}; "
        f"r_max={r_max}, r_min={r_min}, k={k}",
        flush=True,
    )
    _stream_frames(
        args.minus_output,
        cases,
        lambda case: response.retrieve_constant_shear(
            case=case, shear=minus, r_max=r_max, r_min=r_min, k=k,
            real="real0", tile_name=tile_name, data_path=data_path,
            shape_suffix=shape_suffix, include_isolated=include_isolated,
        ),
        finite_column=None,
        label="minus-shear",
    )
    _stream_frames(
        args.response_output,
        cases,
        lambda case: response.retrieve_constant_response(
            case=case, shear_cases=[plus, minus], r_max=r_max, r_min=r_min, k=k,
            real="real0", tile_name=tile_name, data_path=data_path,
            shape_suffix=shape_suffix, include_isolated=include_isolated,
        ),
        finite_column="response",
        label="paired-response",
    )

    print("scanning response means", flush=True)
    new_stats = _accumulate_catalogue(
        args.response_output, start, stop - 1, args.mag_max, args.re_min,
    )
    old_stats = _accumulate_catalogue(
        args.reference_catalogue, args.reference_case_min, args.reference_case_max,
        args.mag_max, args.re_min,
    )
    old_domain = old_stats["domain"]
    new_domain = new_stats["domain"]
    delta = new_domain["case_weighted_mean"] - old_domain["case_weighted_mean"]
    delta_sem = math.hypot(new_domain["case_sem"], old_domain["case_sem"])
    comparison = {
        "delta_new_minus_old": delta,
        "delta_sem_independent_cases": delta_sem,
        "z_independent_cases": delta / delta_sem,
        "fractional_difference": (
            new_domain["case_weighted_mean"] / old_domain["case_weighted_mean"] - 1.0
        ),
    }
    payload = {
        "design": "independent case blocks; case is the uncertainty unit",
        "certified_selection": {
            "r_input_p": [SELECTION_MAG_MIN, SELECTION_MAG_MAX],
            "Re_input_p": [SELECTION_RE_MIN, SELECTION_RE_MAX],
            "distance_for_neighbored": [SELECTION_DISTANCE_MIN, SELECTION_DISTANCE_MAX],
            "isolated_rows_kept": True,
            "intervals": "strict/open",
        },
        "domain": {"r_input_p_max": args.mag_max, "Re_input_p_min": args.re_min},
        "new": {
            "catalogue": str(Path(args.response_output).resolve()),
            "shear_magnitude": expected_mag,
            **new_stats,
        },
        "old": {
            "catalogue": str(Path(args.reference_catalogue).resolve()),
            "shear_magnitude": 0.02,
            **old_stats,
        },
        "domain_comparison": comparison,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = Path(str(summary_path) + ".building")
    if temporary_summary.exists():
        raise FileExistsError(f"refusing to overwrite existing {temporary_summary}")
    with temporary_summary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_summary, summary_path)

    print(
        "DOMAIN RESULT: "
        f"new={new_domain['case_weighted_mean']:.8f} +/- {new_domain['case_sem']:.8f}, "
        f"old={old_domain['case_weighted_mean']:.8f} +/- {old_domain['case_sem']:.8f}, "
        f"delta={delta:+.8f} +/- {delta_sem:.8f} "
        f"(z={comparison['z_independent_cases']:+.2f})",
        flush=True,
    )
    print(f"summary={summary_path}\nCONSTANT_TWO_SIDED_FINISH_DONE", flush=True)


if __name__ == "__main__":
    main()

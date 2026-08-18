"""Build a response catalogue with pair-specific absolute other-scene flux.

For every primary--secondary response row, the three cached full-scene shell
sums are joined by ``(case,input_index)``.  The designated secondary is then
subtracted from its own shell, leaving every other rendered input galaxy.  The
new coordinates are ``log10(1 + F_other)`` in simulation-count units for
0--1, 1--3 and 3--10 arcsec.  Labels and base pair columns are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from blendemu.scene_features import (
    PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS,
    pair_other_absolute_flux,
)


LOOKUP_COLUMNS = (
    "logflux_near_0_1",
    "logflux_mid_1_3",
    "logflux_far_3_10",
)
BASE_COLUMNS = (
    "case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1",
)
POSITION_COLUMNS = ("RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s")


def composite_key(case, input_index):
    case = np.asarray(case, dtype=np.uint64)
    index = np.asarray(input_index, dtype=np.uint64)
    if np.any(case >= 2**32) or np.any(index >= 2**32):
        raise ValueError("case/input_index exceed the 32-bit composite-key allocation")
    return (case << np.uint64(32)) | index


def load_lookup(paths):
    keys, features, provenance = [], [], []
    for path in paths:
        table = pf.read_table(path, columns=["case", "input_index", *LOOKUP_COLUMNS])
        key = composite_key(table["case"].to_numpy(), table["input_index"].to_numpy())
        value = np.column_stack([
            table[name].to_numpy(zero_copy_only=False) for name in LOOKUP_COLUMNS
        ]).astype(np.float32, copy=False)
        order = np.argsort(key, kind="stable")
        key, value = key[order], value[order]
        if len(key) > 1 and np.any(key[1:] == key[:-1]):
            raise ValueError(f"duplicate (case,input_index) keys in {path}")
        keys.append(key); features.append(value)
        provenance.append({"path": os.path.abspath(path), "rows": int(len(key))})
        print(f"lookup {path}: {len(key):,} unique keys", flush=True)
    key = np.concatenate(keys); value = np.concatenate(features)
    order = np.argsort(key, kind="stable")
    key, value = key[order], value[order]
    if len(key) > 1 and np.any(key[1:] == key[:-1]):
        raise ValueError("lookup files overlap in (case,input_index)")
    return key, value, provenance


def match_lookup(keys, values, case, input_index):
    wanted = composite_key(case, input_index)
    position = np.searchsorted(keys, wanted)
    matched = position < len(keys)
    matched[matched] &= keys[position[matched]] == wanted[matched]
    if not matched.all():
        missing = np.flatnonzero(~matched)[:5]
        sample = [(int(np.asarray(case)[i]), int(np.asarray(input_index)[i])) for i in missing]
        raise KeyError(
            f"scene lookup misses {len(matched) - int(matched.sum()):,} rows; sample={sample}"
        )
    return values[position]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--lookups", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-min", type=int, default=0)
    ap.add_argument("--zero-point", type=float, default=30.0)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    lookup_key, lookup_value, lookup_provenance = load_lookup(args.lookups)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = None
    raw_rows = kept_rows = outside_rows = 0
    cases = set()
    with ipc.open_file(args.response) as reader:
        selected = [*BASE_COLUMNS, *POSITION_COLUMNS, "input_index"]
        missing = set(selected) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks columns: {sorted(missing)}")
        try:
            for batch_index in range(reader.num_record_batches):
                table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(selected)
                raw_rows += len(table)
                keep = table["case"].to_numpy(zero_copy_only=False) >= args.case_min
                if not keep.any():
                    continue
                table = table.filter(pa.array(keep))
                case = table["case"].to_numpy(zero_copy_only=False)
                scene = match_lookup(
                    lookup_key, lookup_value, case,
                    table["input_index"].to_numpy(zero_copy_only=False),
                )
                intrinsic_distance = np.hypot(
                    table["RA_input_p"].to_numpy(zero_copy_only=False)
                    - table["RA_input_s"].to_numpy(zero_copy_only=False),
                    table["DEC_input_p"].to_numpy(zero_copy_only=False)
                    - table["DEC_input_s"].to_numpy(zero_copy_only=False),
                ) * 3600.0
                outside_rows += int(np.count_nonzero(
                    (intrinsic_distance < 0.0) | (intrinsic_distance >= 10.0)
                ))
                features = pair_other_absolute_flux(
                    scene,
                    table["r_input_p"].to_numpy(zero_copy_only=False),
                    table["r_input_s"].to_numpy(zero_copy_only=False),
                    intrinsic_distance,
                    zero_point=args.zero_point,
                    allow_outside=True,
                ).astype(np.float32)
                columns = [table[name] for name in BASE_COLUMNS]
                columns.extend(pa.array(features[:, j]) for j in range(3))
                output = pa.Table.from_arrays(
                    columns, names=[*BASE_COLUMNS, *PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS],
                )
                if writer is None:
                    writer = ipc.new_file(args.output, output.schema)
                writer.write_table(output)
                kept_rows += len(output)
                cases.update(np.unique(case).astype(int).tolist())
                print(
                    f"batch {batch_index + 1}/{reader.num_record_batches}: "
                    f"raw={raw_rows:,} kept={kept_rows:,} "
                    f"intrinsic-secondary-outside10={outside_rows:,}", flush=True,
                )
        finally:
            if writer is not None:
                writer.close()
    if writer is None:
        raise RuntimeError("no rows passed case-min")

    metadata = {
        "output": os.path.abspath(args.output),
        "source_response": os.path.abspath(args.response),
        "lookup_inputs": lookup_provenance,
        "raw_rows": int(raw_rows), "rows": int(kept_rows),
        "case_min": int(args.case_min), "case_lo": min(cases), "case_hi": max(cases),
        "n_cases": len(cases), "features": list(PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS),
        "intrinsic_secondary_outside_10_rows": int(outside_rows),
        "pair_shell_separation": "Euclidean intrinsic RA/DEC separation in arcsec",
        "zero_point": float(args.zero_point),
        "feature_definition": (
            "log10(1 + absolute intrinsic flux in simulation counts of every rendered "
            "input source except the primary and this row's designated secondary), "
            "in 0--1/1--3/3--10 arcsec shells"
        ),
    }
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.output}: {kept_rows:,} rows over {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()

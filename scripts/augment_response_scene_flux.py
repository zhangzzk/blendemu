"""Build a compact BlendEMU response catalogue with intrinsic scene context.

The source response labels are unchanged.  Three per-primary features are
strictly joined from precomputed, one-row-per-input lookups:

  * 0--1 arcsec total neighbour flux / primary flux
  * 1--3 arcsec total neighbour flux / primary flux
  * 3--10 arcsec total neighbour flux / primary flux

All fluxes come from rendered input galaxies.  No detection, measurement,
model prediction, or constant-shear validation quantity is read.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from blendemu.scene_features import SCENE_FLUX_PAIR_COLUMNS


LOOKUP_COLUMNS = (
    "logflux_near_0_1",
    "logflux_mid_1_3",
    "logflux_far_3_10",
)
BASE_COLUMNS = (
    "case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1",
)


def composite_key(case, input_index):
    case = np.asarray(case, dtype=np.uint64)
    index = np.asarray(input_index, dtype=np.uint64)
    if np.any(case >= 2**32) or np.any(index >= 2**32):
        raise ValueError("case/input_index exceed the 32-bit composite-key allocation")
    return (case << np.uint64(32)) | index


def load_lookup(paths):
    keys = []
    features = []
    provenance = []
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
        keys.append(key)
        features.append(value)
        provenance.append({"path": os.path.abspath(path), "rows": int(len(key))})
        print(f"lookup {path}: {len(key):,} unique keys", flush=True)
    key = np.concatenate(keys)
    value = np.concatenate(features)
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
        raise KeyError(f"scene lookup misses {len(matched) - int(matched.sum()):,} rows; sample={sample}")
    return values[position]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--lookups", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-min", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")

    lookup_key, lookup_value, lookup_provenance = load_lookup(args.lookups)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    writer = None
    raw_rows = kept_rows = 0
    cases = set()
    with ipc.open_file(args.response) as reader:
        available = set(reader.schema.names)
        required = set(BASE_COLUMNS) | {"input_index"}
        missing = required - available
        if missing:
            raise KeyError(f"response catalogue lacks columns: {sorted(missing)}")
        selected = [*BASE_COLUMNS, "input_index"]
        try:
            for batch_index in range(reader.num_record_batches):
                table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(selected)
                raw_rows += len(table)
                case = table["case"].to_numpy(zero_copy_only=False)
                keep = case >= args.case_min
                if not keep.any():
                    continue
                table = table.filter(pa.array(keep))
                case = table["case"].to_numpy(zero_copy_only=False)
                input_index = table["input_index"].to_numpy(zero_copy_only=False)
                scene = match_lookup(lookup_key, lookup_value, case, input_index)
                columns = [table[name] for name in BASE_COLUMNS]
                columns.extend(pa.array(scene[:, j]) for j in range(3))
                output = pa.Table.from_arrays(columns, names=[*BASE_COLUMNS, *SCENE_FLUX_PAIR_COLUMNS])
                if writer is None:
                    writer = ipc.new_file(args.output, output.schema)
                writer.write_table(output)
                kept_rows += len(output)
                cases.update(np.unique(case).astype(int).tolist())
                print(
                    f"batch {batch_index + 1}/{reader.num_record_batches}: "
                    f"raw={raw_rows:,} kept={kept_rows:,}", flush=True,
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
        "n_cases": len(cases), "features": list(SCENE_FLUX_PAIR_COLUMNS),
        "feature_definition": (
            "log10(1 + total flux of every other rendered input source in the stated "
            "annulus / primary input flux); intrinsic inputs only"
        ),
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"wrote {args.output}: {kept_rows:,} rows over {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare full truth-magnitude-bounded response labels with exact baseline-output joins.

No measured science cut is applied. The g=0 primary must have usable shape,
positive radius/flux, and a valid response label in both shear legs. Baseline
radius and magnitude are optional emulator inputs, never fit-target offsets.

The truth bound is ``--truth-magnitude-max``, defaulting to the historical 26.0
and named to match ``prepare_full_domain_flow.py`` so the emulator and the flow
can be prepared on the same declared truth domain.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from blendemu.data_utils import rescale

FEATURES = ["Re_input_p_scaled", "Re_input_s_scaled", "r_input_p_scaled", "r_input_s_scaled",
            "sersic_n_input_p", "sersic_n_input_s", "distance_scaled"]
OUTPUT_FEATURES = ["baseline_log_flux_radius", "baseline_mag_auto"]
NEED = ["case", "input_index", "shear_angle", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
        "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1", "delta_et2",
        "measured_e1_m", "measured_e2_m", "measured_e1_p", "measured_e2_p"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def baseline_rows(root, case):
    cat = root / f"case{case}_0.0/real0/catalogues"
    cross_path = cat / "CrossMatch/tile180.0_-0.5_rot0_matched.feather"
    shape_path = cat / "Shapes/shape_catalogue_detect_position_tile180.0_-0.5.feather"
    cross = pd.read_feather(cross_path, columns=["id_detec", "id_input"])
    shape = pd.read_feather(shape_path, columns=["NUMBER", "NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"])
    if (cross["id_detec"].duplicated().any() or cross["id_input"].duplicated().any()
            or shape["NUMBER"].duplicated().any()):
        raise ValueError("nonunique baseline identities")
    with ipc.open_file(root / f"gals{case}_0.2.feather") as reader:
        n_generated = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
    joined = cross.merge(shape, left_on="id_detec", right_on="NUMBER", how="inner", validate="one_to_one")
    values = joined[["NGMIX_G1", "NGMIX_G2", "MAG_AUTO", "FLUX_RADIUS"]].to_numpy(float)
    valid = np.isfinite(values).all(1) & (np.square(values[:, :2]).sum(1) < 1) & (values[:, 3] > 0)
    valid &= (joined["id_input"].to_numpy() >= 0) & (joined["id_input"].to_numpy() < n_generated // 2)
    result = joined.loc[valid].set_index("id_input")
    return result, {"raw_cross": len(cross), "cross_without_shape_dropped": len(cross)-len(joined),
                    "usable_primary_baseline": len(result), "n_generated": n_generated,
                    "cross_sha256": sha256(cross_path), "shape_sha256": sha256(shape_path)}


def join_and_select(frame, baseline, truth_magnitude_max):
    """Report all drops and reject a label/baseline measurement mismatch."""
    raw = len(frame)
    truth = np.isfinite(frame["r_input_p"]) & (frame["r_input_p"] < truth_magnitude_max)
    frame = frame.loc[truth].copy()
    finite = np.isfinite(frame[NEED].to_numpy(float)).all(1)
    values = frame[["measured_e1_m", "measured_e2_m", "measured_e1_p", "measured_e2_p"]].to_numpy(float)
    valid = finite & (np.square(values[:, :2]).sum(1) < 1) & (np.square(values[:, 2:]).sum(1) < 1)
    label_drops = int((~valid).sum())
    frame = frame.loc[valid]
    aligned = baseline.reindex(frame["input_index"])
    present = aligned["MAG_AUTO"].notna().to_numpy()
    missing = int((~present).sum())
    frame, aligned = frame.loc[present].copy(), aligned.loc[present]
    if not np.allclose(frame[["measured_e1_m", "measured_e2_m"]].to_numpy(),
                       aligned[["NGMIX_G1", "NGMIX_G2"]].to_numpy(), rtol=0, atol=1e-7):
        raise ValueError("response label's g0 shape differs from joined baseline")
    frame[OUTPUT_FEATURES[0]] = np.log(aligned["FLUX_RADIUS"].to_numpy(float))
    frame[OUTPUT_FEATURES[1]] = aligned["MAG_AUTO"].to_numpy(float)
    counts = {"raw_rows": raw, "truth_cut_dropped": int((~truth).sum()),
              "invalid_label_dropped": label_drops, "baseline_unavailable_dropped": missing, "kept_rows": len(frame)}
    assert sum(counts[k] for k in counts if k != "raw_rows") == raw
    return frame, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-stop", type=int, default=200)
    parser.add_argument("--truth-magnitude-max", type=float, default=26.0)
    # Defaults are the FS2 conditions; other image sims must pass their own.
    parser.add_argument("--pixel-rms", type=float, default=.312)
    parser.add_argument("--psf-fwhm", type=float, default=.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    args = parser.parse_args()
    if not np.isfinite(args.truth_magnitude_max):
        raise ValueError("--truth-magnitude-max must be finite")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("preparation requires Slurm")
    RESCALE = {"pixel_rms": args.pixel_rms, "pixel_size": .2, "zero_mag": 30.,
               "psf_fwhm": args.psf_fwhm, "moffat_beta": args.moffat_beta}
    args.output_root.mkdir(parents=True, exist_ok=False)
    parts_root = args.output_root / "parts"
    parts_root.mkdir()
    source = args.source_root / "response_catalogue_train.feather"
    reports, baseline_reports, parts = {}, {}, []
    @lru_cache(maxsize=4)
    def load_baseline(case):
        rows, report = baseline_rows(args.source_root, case)
        baseline_reports[str(case)] = report
        return rows
    fit_count, fit_sum, fit_square = 0, 0., 0.
    with ipc.open_file(source) as reader:
        if not set(NEED) <= set(reader.schema.names):
            raise ValueError("response catalogue schema mismatch")
        for bi in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(bi)]).select(NEED).to_pandas()
            for case, block in frame.groupby("case", sort=False):
                case = int(case)
                if not 0 <= case < args.case_stop:
                    continue
                selected, counts = join_and_select(
                    block, load_baseline(case), args.truth_magnitude_max)
                total = reports.setdefault(str(case), {key: 0 for key in counts})
                for key, value in counts.items():
                    total[key] += value
                if selected.empty:
                    continue
                scaled = rescale(selected.copy(), **RESCALE)
                x = scaled[FEATURES + OUTPUT_FEATURES].to_numpy(np.float32)
                y = (selected["delta_et1"].to_numpy(float) / .2).astype(np.float32)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("nonfinite prepared features or labels")
                name = f"part{bi:05d}_case{case:03d}.npz"
                path = parts_root / name
                np.savez(path, x=x, y=y, primary=selected["input_index"].to_numpy(np.int64),
                         shear_angle=selected["shear_angle"].to_numpy(np.float32),
                         null=(selected["delta_et2"].to_numpy(float)/.2).astype(np.float32))
                parts.append({"path": f"parts/{name}", "case": case, "rows": len(y), "sha256": sha256(path)})
                if case >= 40:
                    fit_count += len(y)
                    fit_sum += y.astype(float).sum()
                    fit_square += np.square(y.astype(float)).sum()
            if bi % 100 == 0:
                print(f"OUTPUT_RESPONSE_PREP batch={bi}/{reader.num_record_batches} parts={len(parts)}", flush=True)
    if sorted(map(int, reports)) != list(range(args.case_stop)) or not fit_count:
        raise ValueError("incomplete case coverage or empty training sample")
    mean = fit_sum / fit_count
    std = np.sqrt((fit_square - fit_count * mean**2) / (fit_count - 1))
    write_json(args.output_root / "manifest.json", {
        "source": str(source), "source_sha256": sha256(source), "script_sha256": sha256(__file__),
        "features": FEATURES + OUTPUT_FEATURES, "rescale": RESCALE, "shear": .2,
        "fit_cases": [40, args.case_stop-1], "development_cases": [0, 39],
        "fit_count": fit_count, "y_mean": float(mean), "y_std": float(std),
        "counts": reports, "baseline_sources": baseline_reports, "parts": parts,
        "truth_magnitude_max": float(args.truth_magnitude_max),
        "population": f"structural primary target; true r<{args.truth_magnitude_max:g}; usable g0 shape and positive radius; both response-label shapes valid; no measured science cut",
        "limitations": ["forward response labels at h=0.2", "requires a valid g0 output", "development scenes previously used in other models"],
    })
    print(f"OUTPUT_RESPONSE_PREP_COMPLETE fit_pairs={fit_count} parts={len(parts)}", flush=True)


if __name__ == "__main__":
    main()

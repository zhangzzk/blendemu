"""Rebuild the constant-shear (constgold) response catalogue WITH per-leg measured mag and size.

WHY. Until now the only per-leg measured quantity in this catalogue was ``S/N``. That forced any
selection test to cut the MODEL on a mag+size PROXY for S/N -- the flow does not output
``FLUX_AUTO``/``FLUXERR_AUTO``, so it cannot form a real S/N -- and that proxy carried an error of
its own (-0.16 to -0.26 pts at mild cuts, measured on the half-shear legs). ``MAG_AUTO`` and
``FLUX_RADIUS`` are plain SExtractor outputs already present in the shape catalogue the builder
loads, so writing them out removes the proxy from the constgold selection test entirely: the flow
outputs measured magnitude and size directly, and sim and model can cut on the SAME real quantity.

WHY THIS SCRIPT EXISTS INSTEAD OF `run_pipeline.py step_constant_catalogs`. That step calls
`_remove_outputs` on the response catalogue BEFORE rebuilding, i.e. it DELETES the certified
evaluation file first. constgold is evaluation-only and there is a documented history of stale-vs-fixed
catalogue versions producing wrong numbers, so this writes to a SEPARATE path and touches nothing
that exists. Promotion is a later, explicit decision.

IT ALSO RE-RUNS THE SHAPE-CATALOGUE STEP: no. Only `retrieve_constant_response` is called, which
reads existing per-case simulation outputs. No simulation, no shape measurement, no training.

VERIFICATION IS THE POINT, NOT A FORMALITY. `--verify-against` reloads the certified catalogue and
checks that every PRE-EXISTING column is reproduced exactly (same row count, same order, same values
including NaN placement). If a single old column differs, the neighbour parameters or the source data
have drifted and the new file must NOT be used -- the run exits non-zero and says so.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

BE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import response  # noqa: E402
from blendemu import catalog  # noqa: E402
from blendemu.config import load_config  # noqa: E402

NEW_COLS = ["measured_mag_auto_plus", "measured_mag_auto_minus",
            "measured_flux_radius_plus", "measured_flux_radius_minus"]


def build(cfg, out_prefix, cases, n_jobs, tmp_dir, drop_shape_failures):
    from joblib import Parallel, delayed

    sim_cfg, cat_cfg, sm_cfg = cfg["simulation"], cfg["catalogues"], cfg["shape_measurement"]
    cs = cat_cfg["constant_shear"]
    out_path = sim_cfg["output_path"]
    g_mag = abs(float(sim_cfg["constant_shear"]))
    plus_label = catalog.shear_label(g_mag)
    minus_label = catalog.shear_label(-g_mag)
    print(f"  legs: +{plus_label} / {minus_label}   r_max={cs['r_max']}\" r_min={cs.get('r_min',0)} "
          f"k={cs['k']} include_isolated={cs.get('include_isolated', True)}", flush=True)

    def one(case):
        try:
            df = response.retrieve_constant_response(
                case=case, shear_cases=[plus_label, minus_label],
                r_max=cs["r_max"], r_min=cs.get("r_min", 0), k=cs["k"], real="real0",
                tile_name=sm_cfg["tile_name"], data_path=out_path,
                shape_suffix=cs.get("shape_suffix", "_all"),
                include_isolated=bool(cs.get("include_isolated", True)),
            )
        except FileNotFoundError as e:
            print(f"  case {case}: MISSING {e}", flush=True)
            return None
        if drop_shape_failures:
            # NOW REDUNDANT, KEPT AS A GUARD. `retrieve_constant_response` used to MARK ngmix shape
            # failures NaN and KEEP the rows, which is how the first rebuild came out +5,313 rows
            # (+0.013%) against the reference -- exactly the NaN-shape count. That drop has since
            # been restored inside the builder itself, so this normally removes 0 rows; it stays so
            # the driver still reproduces the reference if run against an older blendemu.
            keep = np.isfinite(df["measured_e1_plus"].to_numpy(float)) \
                & np.isfinite(df["measured_e1_minus"].to_numpy(float))
            df = df.loc[keep].reset_index(drop=True)
        p = os.path.join(tmp_dir, f"_part_{case:04d}.feather")
        df.to_feather(p)
        return p

    t0 = time.time()
    paths = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(c) for c in cases)
    paths = [p for p in paths if p]
    print(f"  built {len(paths)}/{len(cases)} cases ({time.time()-t0:.0f}s)", flush=True)
    if len(paths) != len(cases):
        raise SystemExit(f"REFUSING TO MERGE: {len(cases)-len(paths)} cases failed to build. "
                         "A partial catalogue would silently change the evaluation population.")

    # Merge in CASE ORDER so row order matches the original merge (which concatenated batches in
    # ascending case order). Row order must match for the verification below to be meaningful.
    parts = [pd.read_feather(p) for p in sorted(paths)]
    whole = pd.concat(parts, ignore_index=True)
    out = f"{out_prefix}_train.feather"
    whole.to_feather(out)
    print(f"  merged {whole.shape[0]:,} rows x {whole.shape[1]} cols -> {out}", flush=True)
    for p in paths:
        os.remove(p)
    return out


def verify(new_path, ref_path):
    print("\n" + "=" * 96)
    print("VERIFY: every PRE-EXISTING column must be reproduced exactly")
    print("=" * 96)
    ref = pd.read_feather(ref_path)
    new = pd.read_feather(new_path)
    print(f"  reference: {ref.shape[0]:,} rows x {ref.shape[1]} cols   {ref_path}")
    print(f"  new      : {new.shape[0]:,} rows x {new.shape[1]} cols   {new_path}")

    ok = True
    if ref.shape[0] != new.shape[0]:
        print(f"  FAIL: row count {ref.shape[0]:,} -> {new.shape[0]:,}")
        return False
    missing = [c for c in ref.columns if c not in new.columns]
    if missing:
        print(f"  FAIL: columns dropped: {missing}")
        ok = False
    added = [c for c in new.columns if c not in ref.columns]
    print(f"  columns added: {added}")
    if sorted(added) != sorted(NEW_COLS):
        print(f"  FAIL: expected exactly {sorted(NEW_COLS)}")
        ok = False

    bad = []
    for c in ref.columns:
        if c not in new.columns:
            continue
        a, b = ref[c].to_numpy(), new[c].to_numpy()
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        if not same:
            n_diff = int(np.sum(a != b)) if a.dtype == b.dtype else -1
            bad.append((c, n_diff))
    if bad:
        ok = False
        print(f"  FAIL: {len(bad)} pre-existing column(s) changed:")
        for c, n in bad[:20]:
            print(f"    {c:<34} n_differing={n}")
    else:
        print(f"  PASS: all {len(ref.columns)} pre-existing columns identical (NaN placement included)")

    for c in NEW_COLS:
        if c in new.columns:
            v = new[c].to_numpy(float)
            f = np.isfinite(v)
            print(f"  {c:<32} finite={100*f.mean():5.1f}%  "
                  f"median={np.median(v[f]) if f.any() else np.nan:.4f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(BE, "configs/fs2_lsst_r_constant_g002.yaml"))
    ap.add_argument("--out-prefix", required=True,
                    help="written as <prefix>_train.feather; must NOT be the certified path")
    ap.add_argument("--n-cases", type=int, default=140)
    ap.add_argument("--case-offset", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--verify-against", default=None)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--keep-shape-failures", action="store_true",
                    help="keep NaN-shape (ngmix-failure) rows. The certified catalogue DROPS them, "
                         "so keeping them will fail --verify-against by ~+0.013%% of rows.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = f"{args.out_prefix}_train.feather"
    if args.verify_against and os.path.abspath(out) == os.path.abspath(args.verify_against):
        raise SystemExit("REFUSING: --out-prefix resolves to the reference file itself.")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} already exists; delete it explicitly if you mean to redo.")

    tmp = args.tmp_dir or os.path.join(os.path.dirname(out), "_rebuild_parts")
    os.makedirs(tmp, exist_ok=True)
    cases = list(range(args.case_offset, args.case_offset + args.n_cases))
    print(f"### REBUILD constgold WITH measured mag+size ###")
    print(f"  config={args.config}\n  cases={cases[0]}..{cases[-1]}  n_jobs={args.n_jobs}")
    print(f"  drop ngmix-failure rows: {not args.keep_shape_failures}")
    new_path = build(cfg, args.out_prefix, cases, args.n_jobs, tmp,
                     drop_shape_failures=not args.keep_shape_failures)

    if args.verify_against:
        if not verify(new_path, args.verify_against):
            print("\nVERIFY FAILED -- do NOT use this catalogue.")
            raise SystemExit(2)
        print("\nVERIFY PASSED -- new catalogue differs from the certified one ONLY by the added "
              "columns.")
    for leftover in glob.glob(os.path.join(tmp, "_part_*.feather")):
        os.remove(leftover)
    print("REBUILD_DONE", flush=True)


if __name__ == "__main__":
    main()

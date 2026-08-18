"""Is the blending emulator biased in the MEAN (not per-object R2) on its OWN data?

The gold coherent-blend under-count could be (a) the emulator under-predicting the per-pair
response, or (b) an unbiased per-pair emulator whose SUM misses collective/coherent blend.
This measures (a) directly on the response catalogue the emulator was trained on:

  y_true = delta_et1 / shear         (measured per-pair blending response)
  y_pred = un-standardized XGBoost regression prediction

Reports mean(pred)/mean(true) globally, by neighbour distance, and -- decisively -- for the
PER-GALAXY SUMMED R_blend = sum over a primary's neighbours (exactly what inference feeds m).
Uses a subset of cases (all pairs of those cases -> complete neighbour sets).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)
from blendemu import data_utils  # noqa

CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
MODELS = os.path.join(BE, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--max-case", type=int, default=9, help="use cases 0..max_case (complete neighbour sets)")
    ap.add_argument("--tag", default="lsst_r", help="emulator tag (lsst_r | lsst_r_extnbr)")
    ap.add_argument("--shear", type=float, default=0.2, help="response render shear scale (ratio is invariant to it)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(MODELS, f"emulator_metadata_{args.tag}.json")))
    reg = meta["tasks"]["regression"]
    feats = reg["features"]; cuts = reg["cuts"]; std = reg["standardization"]
    rc = meta["training"]["rescale"]
    best_trees = reg["metrics"]["best_trees"]
    print(f"regression cuts={cuts}  features={feats}  standardization={std}  best_trees={best_trees}")

    need = ["case", "input_index", "delta_et1", "distance",
            "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
            "sersic_n_input_p", "sersic_n_input_s"]
    parts = []
    with ipc.open_file(args.catalogue) as r:
        avail = set(r.schema.names); cols = [c for c in need if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if b["case"].min() > args.max_case:   # catalogue is case-ordered ascending
                break
            b = b[b["case"] <= args.max_case]
            if len(b):
                parts.append(b)
    ds = pd.concat(parts, ignore_index=True)
    print(f"loaded cases 0..{args.max_case}: {len(ds):,} pairs")

    ds = data_utils.source_select_reg(ds, cuts=cuts).reset_index(drop=True)
    print(f"after regression cuts: {len(ds):,} pairs")
    ds = data_utils.rescale(ds, pixel_rms=rc["pixel_rms"], pixel_size=rc["pixel_size"],
                            zero_mag=rc["zero_mag"], psf_fwhm=rc["psf_fwhm"], moffat_beta=rc["moffat_beta"])

    y_true = (ds["delta_et1"].to_numpy(float) / args.shear)
    model = xgb.Booster(); model.load_model(os.path.join(MODELS, reg["model_file"]))
    dm = xgb.DMatrix(ds[feats])
    y_pred = model.predict(dm, iteration_range=(0, best_trees)) * std["std"] + std["mean"]

    mt, mp = float(np.mean(y_true)), float(np.mean(y_pred))
    print(f"\n=== PER-PAIR (N={len(ds):,}) ===")
    print(f"  mean(true)={mt:+.5f}  mean(pred)={mp:+.5f}  pred/true={mp/mt:.3f}  (resid {mp/mt-1:+.1%})")

    print(f"\n=== PER-PAIR by neighbour distance (arcsec) ===")
    print(f"{'dist bin':>12} {'mean_true':>10} {'mean_pred':>10} {'pred/true':>10} {'N':>10}")
    dist = ds["distance"].to_numpy(float)
    edges = [0, 1, 2, 3, 4, 5, 7, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 1000:
            continue
        t, p = np.mean(y_true[m]), np.mean(y_pred[m])
        print(f"  [{lo:>2},{hi:>2})  {t:>10.4f} {p:>10.4f} {p/t:>10.3f} {int(m.sum()):>10,}")

    # DECISIVE: per-galaxy summed R_blend (what inference feeds m), grouped by (case, primary index)
    g = pd.DataFrame({"case": ds["case"].to_numpy(np.int64),
                      "ii": ds["input_index"].to_numpy(np.int64),
                      "t": y_true, "p": y_pred}).groupby(["case", "ii"]).agg(
        Rt=("t", "sum"), Rp=("p", "sum"), nn=("t", "size")).reset_index()
    Rt, Rp = g["Rt"].to_numpy(), g["Rp"].to_numpy()
    print(f"\n=== PER-GALAXY SUMMED R_blend (N_gal={len(g):,}, <neighbours>={g['nn'].mean():.2f}) ===")
    print(f"  mean R_blend true={np.mean(Rt):+.4f}  pred={np.mean(Rp):+.4f}  pred/true={np.mean(Rp)/np.mean(Rt):.3f} "
          f"(resid {np.mean(Rp)/np.mean(Rt)-1:+.1%})")
    # bootstrap over galaxies for an error bar
    rng = np.random.default_rng(0); rr = []
    idx = np.arange(len(g))
    for _ in range(200):
        s = rng.choice(idx, size=len(idx), replace=True)
        rr.append(np.mean(Rp[s]) / np.mean(Rt[s]) - 1)
    print(f"  pred/true - 1 = {np.mean(Rp)/np.mean(Rt)-1:+.1%} +/- {np.std(rr):.1%} (galaxy bootstrap)")


if __name__ == "__main__":
    main()

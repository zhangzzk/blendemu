"""How much of the gold R_blend under-count is the emulator's DISTANCE-dependent bias?

The emulator is mean-unbiased on its training data (-0.8%), but under-predicts <1" pairs (-11%)
and over-predicts 3-5" pairs (+6%), cancelling in the TRAINING distance mix. If the gold's
detected galaxies weight close pairs more, the -11% no longer cancels -> a real R_blend under-count.

Test: measure r(d) = emulator per-distance bias (pred/true - 1) on the response catalogue;
then for the gold input fields, run the emulator per pair, and form corrected_true(pair) =
response / (1 + r(d_pair)). Compare summed pred vs corrected-true per galaxy -> the emulator's
distance-bias-induced under-count on the GOLD neighbour distribution, globally and by crowding.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf
import xgboost as xgb

BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)
from blendemu import data_utils  # noqa
from blendemu.inference import BlendingPredictor  # noqa

RESP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
MODELS = os.path.join(BE, "models")
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"


def emulator_rd(meta, max_case, shear, edges):
    """r(d) = mean(pred)/mean(true) - 1 per fine distance bin on the response catalogue."""
    reg = meta["tasks"]["regression"]; feats = reg["features"]; cuts = reg["cuts"]
    std = reg["standardization"]; rc = meta["training"]["rescale"]; bt = reg["metrics"]["best_trees"]
    need = ["case", "delta_et1", "distance", "Re_input_p", "Re_input_s",
            "r_input_p", "r_input_s", "sersic_n_input_p", "sersic_n_input_s"]
    parts = []
    with ipc.open_file(RESP) as r:
        cols = [c for c in need if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if b["case"].min() > max_case:
                break
            b = b[b["case"] <= max_case]
            if len(b):
                parts.append(b)
    ds = pd.concat(parts, ignore_index=True)
    ds = data_utils.source_select_reg(ds, cuts=cuts).reset_index(drop=True)
    ds = data_utils.rescale(ds, pixel_rms=rc["pixel_rms"], pixel_size=rc["pixel_size"],
                            zero_mag=rc["zero_mag"], psf_fwhm=rc["psf_fwhm"], moffat_beta=rc["moffat_beta"])
    y_true = ds["delta_et1"].to_numpy(float) / shear
    model = xgb.Booster(); model.load_model(os.path.join(MODELS, reg["model_file"]))
    y_pred = model.predict(xgb.DMatrix(ds[feats]), iteration_range=(0, bt)) * std["std"] + std["mean"]
    dist = ds["distance"].to_numpy(float)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    delta = np.zeros(len(ctr))   # ADDITIVE per-pair deficit mean_true - mean_pred (stable, no division)
    print(f"\ndeficit delta(d)=mean_true-mean_pred from response cat (cases 0..{max_case}, N={len(ds):,}):")
    print(f"{'d':>6} {'mean_true':>10} {'mean_pred':>10} {'delta':>10} {'r(d)':>8} {'N':>10}")
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (dist >= lo) & (dist < hi)
        if m.sum() > 2000:
            t, p = y_true[m].mean(), y_pred[m].mean()
            delta[i] = t - p if abs(t) > 0.003 else 0.0   # zero the noise floor (far pairs ~0 response)
            rr = f"{p/t-1:+.1%}" if abs(t) > 0.003 else "  noise"
            print(f"{ctr[i]:>6.2f} {t:>10.4f} {p:>10.4f} {delta[i]:>+10.4f} {rr:>8} {int(m.sum()):>10,}")
    return ctr, delta


def input_feather(case, sign="0.02"):
    return f"{CBASE}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-cases", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rd-max-case", type=int, default=9)
    ap.add_argument("--shear", type=float, default=0.2)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(MODELS, "emulator_metadata_lsst_r.json")))
    edges = np.arange(0.0, 10.01, 0.5)
    ctr, delta = emulator_rd(meta, args.rd_max_case, args.shear, edges)

    pred = BlendingPredictor.load(MODELS, tag="lsst_r", conditions=COND, device="cpu")
    Rp_all, Rt_all = [], []
    for c in args.gold_cases:
        fp = input_feather(c)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING", flush=True); continue
        t = pf.read_table(fp).to_pandas()
        t = t.rename(columns={col: col.replace("_input", "") for col in t.columns})
        reg = pred.predict_response(t, t)
        d = reg["distance"].to_numpy(float); resp = reg["response"].to_numpy(float)
        di = np.interp(d, ctr, delta, left=delta[0], right=0.0)  # additive deficit at this pair's distance
        corr_true = resp + di                                    # de-biased per-pair "truth" = pred + deficit
        pk = [col for col in reg.columns if col.startswith("index_input_p")][0]
        gg = pd.DataFrame({"ii": reg[pk].to_numpy(np.int64), "p": resp, "t": corr_true}).groupby(
            "ii").agg(Rp=("p", "sum"), Rt=("t", "sum")).reset_index()
        Rp_all.append(gg["Rp"].to_numpy()); Rt_all.append(gg["Rt"].to_numpy())
        print(f"case{c}: {len(gg):,} gals  meanRp={gg['Rp'].mean():.4f} meanRt={gg['Rt'].mean():.4f}", flush=True)
    Rp = np.concatenate(Rp_all); Rt = np.concatenate(Rt_all)

    print(f"\n=== GOLD emulator distance-bias under-count (N_gal={len(Rp):,}) ===")
    print(f"  R_blend  pred={Rp.mean():.4f}  debiased-true={Rt.mean():.4f}  pred/true={Rp.mean()/Rt.mean():.3f} "
          f"(under-count {Rp.mean()/Rt.mean()-1:+.1%})")
    # by R_blend(pred) quantile -> is the under-count concentrated in crowded galaxies?
    q = np.quantile(Rp, [0, 0.5, 0.8, 0.95, 1.0])
    print(f"\n{'Rp quantile':>14} {'meanRp':>8} {'meanRt':>8} {'pred/true':>10} {'N':>9}")
    labs = ["q0-50", "q50-80", "q80-95", "q95-100"]
    for i in range(4):
        m = (Rp >= q[i]) & (Rp <= q[i + 1] if i == 3 else Rp < q[i + 1])
        if m.sum() < 100:
            continue
        print(f"{labs[i]:>14} {Rp[m].mean():>8.4f} {Rt[m].mean():>8.4f} {Rp[m].mean()/Rt[m].mean():>10.3f} {int(m.sum()):>9,}")
    # m-impact if we scale the gold R_blend (0.144) by 1/f and hold R_flow=0.297, R_sim=0.470
    f = Rp.mean() / Rt.mean()
    Rflow, Rsim, Rblend_gold = 0.2966, 0.4698, 0.1442
    m_now = Rsim / (Rflow + Rblend_gold) - 1
    m_corr = Rsim / (Rflow + Rblend_gold / f) - 1
    print(f"\n  m-impact (approx): current +6.6%  ->  with distance-debiased R_blend/f: {m_corr:+.2%} "
          f"(recovers {m_now - m_corr:+.2%} of the residual)")


if __name__ == "__main__":
    main()

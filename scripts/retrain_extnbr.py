"""Retrain the blending-response (regression) emulator on the EXTENDED-NEIGHBOUR domain.

Reuses the OFFICIAL preprocessing (train_emulator.load_regression_data: same rescale,
standardization, train/test split) and the EXACT production hyperparameters from
emulator_metadata_lsst_r.json — the ONLY change is the relaxed secondary cuts in
fs2_lsst_r_extnbr.yaml. So any difference vs production is the domain extension, not tuning.

Saves regression_model_lsst_r_extnbr.json + emulator_metadata_lsst_r_extnbr.json.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

BE = "/home/z/Zekang.Zhang/blendemu"
sys.path.insert(0, BE)
sys.path.insert(0, os.path.join(BE, "scripts"))
from blendemu.config import load_config  # noqa
from blendemu import data_utils  # noqa
import train_emulator as TE  # noqa

# Columns needed for cuts (r/Re _p/_s, distance), rescale (same), features (+sersic_n), target.
_NEED = ["case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
         "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1"]


def load_regression_data_lowmem(cfg):
    """Memory-frugal twin of TE.load_regression_data: batch-read only needed columns and
    apply the cuts per batch (the 27.7GB full-column read OOMs). Post-cut dataframe order,
    standardization, and the fixed-seed split are IDENTICAL to the official loader."""
    tr = cfg["training"]; rc = tr["rescale"]
    cat_path = os.path.join(cfg["simulation"]["output_path"], "response_catalogue_train.feather")
    print(f"Loading (batched, cols={_NEED}): {cat_path}")
    ho_min = int(os.environ.get("HELDOUT_MIN_CASE", "0"))  # train only on case >= this (leakage-free held-out test)
    parts = []; n_raw = 0
    with ipc.open_file(cat_path) as r:
        cols = [c for c in _NEED if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if ho_min > 0:
                b = b[b["case"] >= ho_min]
                if not len(b):
                    continue
            n_raw += len(b)
            parts.append(data_utils.source_select_reg(b, cuts=tr["regression_cuts"]))
    dataset = pd.concat(parts, ignore_index=True); del parts
    if ho_min > 0:
        print(f"  HELD-OUT: trained on case>={ho_min} only (gold cases 0-{ho_min-1} excluded)")
    print(f"  Raw: {n_raw:,} rows;  After cuts: {dataset.shape[0]:,} rows")
    dataset = data_utils.rescale(dataset, pixel_rms=rc["pixel_rms"], pixel_size=rc["pixel_size"],
                                 zero_mag=rc["zero_mag"], psf_fwhm=rc["psf_fwhm"], moffat_beta=rc["moffat_beta"])
    shear = TE._catalogue_shear_scale(cfg, "response")
    y = dataset["delta_et1"] / shear
    y_mean, y_std = float(np.mean(y)), float(np.std(y, ddof=1))
    y = data_utils.standardize(y, y_mean, y_std)
    print(f"  Standardized: mean={y_mean:.6f}, std={y_std:.4f}")
    features = tr["features"]
    # optional close-pair sample weighting (feature-based -> unbiased conditional mean, just
    # re-emphasizes the loss toward the steep <2" close pairs the emulator under-fits at [0,1")=0.90).
    K = float(os.environ.get("WEIGHT_CLOSE", "0"))
    dist = dataset["distance"].to_numpy(float)
    w = (1.0 + K * np.exp(-dist / 1.0)) if K > 0 else np.ones(len(dataset))
    if K > 0:
        print(f"  close-pair weighting: w=1+{K}*exp(-d/1.0)  (w[d=0.5]={1+K*np.exp(-0.5):.2f}, w[d=3]={1+K*np.exp(-3):.2f})")
    Xall = dataset[features].copy(); Xall["_w"] = w
    x_train, x_test, y_train, y_test = train_test_split(
        Xall, y, test_size=tr["test_size"], random_state=tr["random_state"])
    w_train = x_train.pop("_w").to_numpy(float); x_test.pop("_w")
    print(f"  Train: {x_train.shape[0]:,}, Test: {x_test.shape[0]:,}")
    return (xgb.DMatrix(x_train, y_train, weight=w_train), xgb.DMatrix(x_test, y_test),
            x_train, x_test, y_train, y_test, y_mean, y_std)


def main():
    cfg = load_config(os.environ.get("CONFIG_PATH", os.path.join(BE, "configs/fs2_lsst_r_extnbr.yaml")))
    prod = json.load(open(os.path.join(BE, "models/emulator_metadata_lsst_r.json")))
    params = dict(prod["tasks"]["regression"]["params"])
    # device from env (GPU contended -> run on CPU; hist trees are equivalent). Default cpu.
    params.update({"objective": "reg:squarederror", "n_jobs": -1,
                   "device": os.environ.get("XGB_DEVICE", "cpu"), "tree_method": "hist",
                   "booster": "gbtree", "disable_default_eval_metric": 1})
    print(f"reusing production params: {params}")
    print(f"extended regression_cuts: {cfg['training']['regression_cuts']}")

    DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = load_regression_data_lowmem(cfg)

    custom_metric = lambda predt, d: ("r2", r2_score(d.get_label(), predt))
    evals_result = {}
    t0 = time.time()
    bst = xgb.train(params, DMtrain, evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=evals_result, num_boost_round=2000, verbose_eval=100,
                    custom_metric=custom_metric, maximize=True,
                    callbacks=[TE._early_stopping_callback(cfg, "regression", "r2", maximize=True)])
    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  Time: {time.time()-t0:.1f}s  best_iter={bst.best_iteration}  trees={best_trees}  R2={bst.best_score:.6f}")

    model_dir = cfg["training"]["model_dir"]; features = cfg["training"]["features"]
    model_path = TE._fname(model_dir, "regression_model.json", cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    train_curve_path = TE._fname(model_dir, "regression_train_curve.npz", cfg)
    bst.save_model(model_path)
    np.savez(train_curve_path, train_r2=evals_result["train"]["r2"], eval_r2=evals_result["eval"]["r2"])
    meta_path = TE._update_metadata(
        cfg, "regression", model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=train_curve_path,
        metrics={"best_iteration": bst.best_iteration, "best_trees": best_trees,
                 "best_score": bst.best_score, "score_name": "r2"})
    print(f"  saved model={model_path}\n  metadata={meta_path}  standardization=({y_mean:.6f},{y_std:.4f})")
    print(f"EXTNBR_TRAIN_DONE best_trees={best_trees} R2={bst.best_score:.6f}")


if __name__ == "__main__":
    main()

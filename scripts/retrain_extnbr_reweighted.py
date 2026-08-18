"""Reweighted blending-response (R_blend) emulator training and tuning.

Port of the SBSI-side V3 recipe (archive/pre-v3/scripts/tune_v22_reweighted_vector_optuna.py)
onto BlendEMU's own data path: every candidate is TWO sequential trainings on the
official preprocessing (retrain_extnbr.load_regression_data conventions: same cuts,
rescale, frozen source standardization, official random-row split restricted to the
fit cases),

1. fit an ordinary squared-error seven-feature pair model with RMSE early stopping;
2. derive positive-response weights from THAT candidate's own base predictions and
   fit a new model from scratch with the same parameters and the base's tree count.

--mode tune optimizes ``abs(beta - 1)`` with Optuna, where beta is the case-balanced
projection slope of the measured scene response vector on the summed predicted
pair-response vector, evaluated on the case-disjoint half-shear development cases.
--mode train repeats the two-stage fit with a fixed recipe (from a tune summary or
explicit arguments), and can additionally write the standard BlendEMU metadata
sidecar so BlendingPredictor can load the weighted model.

Coherent anchors and ConstGold are never read; they remain possible transfer tests.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import xgboost as xgb
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
for _path in (PACKAGE_ROOT, SCRIPT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blendemu import data_utils  # noqa: E402
from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402

# Columns needed for the case window, cuts, rescale, target, and the vector score.
_NEED = ["case", "input_index", "shear_angle",
         "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
         "sersic_n_input_p", "sersic_n_input_s", "distance",
         "delta_et1", "delta_et2"]

WEIGHT_CAP = 50.0
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
SEED = 20260816
ENQUEUE_ALPHAS = (0.035, 0.05, 0.065, 0.10)
TUNED_PARAM_NAMES = (
    "subsample", "colsample_bytree", "learning_rate", "max_depth",
    "min_child_weight", "reg_lambda", "reg_alpha", "gamma",
)


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        # before the int branch: bool subclasses int and would serialize as 0/1
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def atomic_json(path, payload):
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_model(booster, path):
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    stem, ext = os.path.splitext(os.path.basename(path))
    temporary = os.path.join(os.path.dirname(path), f".{stem}.{os.getpid()}.tmp{ext}")
    booster.save_model(temporary)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stat(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite values")
    return {
        "mean": float(values.mean()),
        "case_sd": float(values.std(ddof=1)),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def r2(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual_ss = float(np.square(target - prediction).sum())
    total_ss = float(np.square(target - target.mean()).sum())
    if total_ss <= 0.0:
        raise RuntimeError("target variance is zero")
    return 1.0 - residual_ss / total_ss


def positive_response_weights(train_prediction, validation_prediction, alpha,
                              cap=WEIGHT_CAP):
    """Feature-only sample weights: (1 + alpha * min(max(pred,0)^2/mean_power, cap)) / mean.

    Labels are never used. The normalization is the TRAIN mean of the raw weight,
    so mean train weight is exactly 1; the validation weight reuses it.
    """
    train_signal = np.maximum(np.asarray(train_prediction, dtype=np.float64), 0.0) ** 2
    validation_signal = np.maximum(np.asarray(validation_prediction, dtype=np.float64), 0.0) ** 2
    mean_power = float(train_signal.mean())
    if not mean_power > 0.0:
        raise RuntimeError("base prediction has no positive response power")
    train_ratio = np.minimum(train_signal / mean_power, cap)
    validation_ratio = np.minimum(validation_signal / mean_power, cap)
    normalization = float((1.0 + alpha * train_ratio).mean())
    train_weight = ((1.0 + alpha * train_ratio) / normalization).astype(np.float32)
    validation_weight = ((1.0 + alpha * validation_ratio) / normalization).astype(np.float32)
    metadata = {
        "alpha": float(alpha),
        "cap": float(cap),
        "train_mean_prediction_power": mean_power,
        "raw_weight_normalization": normalization,
        "train_weight_p50": float(np.quantile(train_weight, 0.50)),
        "train_weight_p99": float(np.quantile(train_weight, 0.99)),
        "train_weight_max": float(train_weight.max()),
    }
    return train_weight, validation_weight, metadata


class VectorEvaluation:
    """Fast reconstruction and scoring of half-shear scene-response vectors.

    beta = case-balanced projection slope of the measured scene response vector on
    the summed predicted pair-response vector (spin-2 convention: phase 2*theta).
    """

    def __init__(self, case, primary, shear_angle_degree, label, null):
        self.case = np.asarray(case, dtype=np.int16)
        self.primary = np.asarray(primary, dtype=np.int64)
        phase = np.deg2rad(2.0 * np.asarray(shear_angle_degree, dtype=np.float64))
        self.cosine = np.cos(phase).astype(np.float32)
        self.sine = np.sin(phase).astype(np.float32)
        key = (self.case.astype(np.int64) << np.int64(48)) + self.primary
        if np.any(key[1:] < key[:-1]):
            raise RuntimeError("evaluation rows are not ordered by case/primary")
        change = np.empty(len(key), dtype=bool)
        change[0] = True
        change[1:] = key[1:] != key[:-1]
        self.starts = np.flatnonzero(change).astype(np.int64)
        self.scene_case = self.case[self.starts]
        csum = np.add.reduceat(self.cosine.astype(np.float64), self.starts)
        ssum = np.add.reduceat(self.sine.astype(np.float64), self.starts)
        lsum = np.add.reduceat(np.asarray(label, dtype=np.float64), self.starts)
        nsum = np.add.reduceat(np.asarray(null, dtype=np.float64), self.starts)
        direction_power = np.square(csum) + np.square(ssum)
        if np.any(direction_power <= 1.0e-8):
            raise RuntimeError("encountered vanishing summed shear direction")
        self.measured1 = (lsum * csum - nsum * ssum) / direction_power
        self.measured2 = (lsum * ssum + nsum * csum) / direction_power

    def score(self, pair_prediction, case_window):
        prediction = np.asarray(pair_prediction, dtype=np.float64)
        if prediction.shape != self.cosine.shape:
            raise ValueError("pair prediction length differs from evaluation rows")
        pred1 = np.add.reduceat(prediction * self.cosine, self.starts)
        pred2 = np.add.reduceat(prediction * self.sine, self.starts)
        power = np.square(pred1) + np.square(pred2)
        dot = pred1 * self.measured1 + pred2 * self.measured2
        cross = -pred2 * self.measured1 + pred1 * self.measured2
        case_min, case_max = case_window
        use = (self.scene_case >= case_min) & (self.scene_case <= case_max)
        local_case = self.scene_case[use].astype(np.int64) - case_min
        n_cases = case_max - case_min + 1
        case_power = np.bincount(local_case, weights=power[use], minlength=n_cases)
        case_dot = np.bincount(local_case, weights=dot[use], minlength=n_cases)
        case_cross = np.bincount(local_case, weights=cross[use], minlength=n_cases)
        if np.any(case_power <= 0.0):
            raise RuntimeError("evaluation case has zero predicted vector power")
        slopes = case_dot / case_power
        nulls = case_cross / case_power
        return {
            "case_window": [case_min, case_max],
            "n_pairs": int(((self.case >= case_min) & (self.case <= case_max)).sum()),
            "n_primaries": int(use.sum()),
            "slope_measured_on_predicted": stat(slopes),
            "slope_minus_one": stat(slopes - 1.0),
            "orthogonal_slope_null": stat(nulls),
            "pooled_slope": float(case_dot.sum() / case_power.sum()),
        }


def load_pair_population(cfg, case_min, case_max):
    """Batched twin of the official loader that keeps the vector-score columns.

    Selection order (window, then cuts, then finite check), the official rescale,
    and the per-batch row order are IDENTICAL to the SBSI V3 cache builder, so the
    official random-row split over the fit rows reproduces the same mask.
    """
    tr = cfg["training"]
    cat_path = os.path.join(cfg["simulation"]["output_path"], "response_catalogue_train.feather")
    print(f"Loading (batched, cols={_NEED}): {cat_path}")
    shear = TE._catalogue_shear_scale(cfg, "response")
    if not np.isfinite(shear) or shear <= 0.0:
        raise RuntimeError(f"invalid response shear scale {shear}")
    features = list(tr["features"])
    parts = {name: [] for name in ("case", "primary", "shear_angle", "label", "null")}
    parts["x"] = []
    n_raw = 0
    with ipc.open_file(cat_path) as reader:
        missing = set(_NEED) - set(reader.schema.names)
        if missing:
            raise KeyError(f"response catalogue lacks {sorted(missing)}")
        for bi in range(reader.num_record_batches):
            frame = pa.Table.from_batches([reader.get_batch(bi)]).select(_NEED).to_pandas()
            case = frame.case.to_numpy(np.int64, copy=False)
            frame = frame.loc[(case >= case_min) & (case <= case_max)]
            if frame.empty:
                continue
            frame = data_utils.source_select_reg(frame, cuts=tr["regression_cuts"])
            if frame.empty:
                continue
            n_raw += len(frame)
            numeric = frame[_NEED].to_numpy(np.float64, copy=False)
            if not np.isfinite(numeric).all():
                bad = int((~np.isfinite(numeric)).any(axis=1).sum())
                raise RuntimeError(f"selected batch contains {bad} non-finite rows")
            scaled = data_utils.rescale(
                frame.copy(),
                pixel_rms=tr["rescale"]["pixel_rms"], pixel_size=tr["rescale"]["pixel_size"],
                zero_mag=tr["rescale"]["zero_mag"], psf_fwhm=tr["rescale"]["psf_fwhm"],
                moffat_beta=tr["rescale"]["moffat_beta"])
            parts["case"].append(frame.case.to_numpy(np.int16))
            parts["primary"].append(frame.input_index.to_numpy(np.int64))
            parts["shear_angle"].append(frame.shear_angle.to_numpy(np.float32))
            parts["label"].append((frame.delta_et1.to_numpy(np.float64) / shear).astype(np.float32))
            parts["null"].append((frame.delta_et2.to_numpy(np.float64) / shear).astype(np.float32))
            parts["x"].append(scaled[features].to_numpy(np.float32))
            if (bi + 1) % 100 == 0:
                print(f"  batch {bi + 1}/{reader.num_record_batches}: selected={n_raw:,}", flush=True)
    if n_raw == 0:
        raise RuntimeError("no supported rows in the requested case window")
    population = {
        "case": np.concatenate(parts["case"]),
        "primary": np.concatenate(parts["primary"]),
        "shear_angle": np.concatenate(parts["shear_angle"]),
        "label": np.concatenate(parts["label"]),
        "null": np.concatenate(parts["null"]),
        "x": np.concatenate(parts["x"], axis=0),
        "features": features,
        "shear": float(shear),
        "n_rows": int(n_raw),
    }
    print(f"  Raw in window: selected rows: {n_raw:,}")
    return population


class ReweightedExperiment:
    """Holds the frozen split/matrices and performs the two-stage candidate fit."""

    def __init__(self, population, fit_cases, dev_cases, target_mean, target_std,
                 test_size, random_state):
        case = population["case"].astype(np.int64)
        fit = (case >= fit_cases[0]) & (case <= fit_cases[1])
        dev = (case >= dev_cases[0]) & (case <= dev_cases[1])
        leftover = ~(fit | dev)
        if leftover.any():
            stray = np.unique(case[leftover]).tolist()
            raise RuntimeError(f"rows outside fit/development windows for cases {stray}")
        fit_index = np.flatnonzero(fit)
        local = np.arange(len(fit_index), dtype=np.int32)
        official_train, _ = train_test_split(
            local, test_size=float(test_size), random_state=int(random_state))
        official_mask = np.zeros(len(case), dtype=bool)
        official_mask[fit_index[official_train]] = True
        train = fit & official_mask
        validation = fit & ~official_mask
        self.fit_cases = tuple(fit_cases)
        self.dev_cases = tuple(dev_cases)
        self.features = list(population["features"])
        self.x_train = population["x"][train]
        self.x_validation = population["x"][validation]
        self.x_dev = population["x"][dev]
        self.y_train_physical = population["label"][train]
        self.y_validation_physical = population["label"][validation]
        self.target_mean = float(target_mean)
        self.target_scale = float(target_std)
        self.y_train = ((self.y_train_physical - self.target_mean)
                        / self.target_scale).astype(np.float32)
        self.y_validation = ((self.y_validation_physical - self.target_mean)
                             / self.target_scale).astype(np.float32)
        self.dm_train = xgb.DMatrix(self.x_train, label=self.y_train, feature_names=self.features)
        self.dm_validation = xgb.DMatrix(self.x_validation, label=self.y_validation,
                                         feature_names=self.features)
        self.dm_dev = xgb.DMatrix(self.x_dev, feature_names=self.features)
        self.vector = VectorEvaluation(
            case=case[dev], primary=population["primary"][dev],
            shear_angle_degree=population["shear_angle"][dev],
            label=population["label"][dev], null=population["null"][dev])
        self.n_train = int(train.sum())
        self.n_validation = int(validation.sum())
        self.n_dev = int(dev.sum())
        print(f"  two-stage population: train={self.n_train:,} "
              f"validation={self.n_validation:,} development={self.n_dev:,}")

    def physical_predict(self, booster, matrix, n_trees):
        standardized = booster.predict(matrix, iteration_range=(0, n_trees))
        return (standardized.astype(np.float64) * self.target_scale
                + self.target_mean).astype(np.float32)

    def fit_candidate(self, params, alpha, *, fixed_trees=None, keep_models=False):
        """Two sequential trainings: ordinary base, then its positive-weighted refit.

        fixed_trees pins both stages to an exact tree count with no early stopping
        (the fixed-recipe transfer variant); otherwise the base early-stops on
        validation RMSE and the refit reuses the selected tree count.
        """
        fit_params = common_params()
        fit_params.update(params)
        started = time.time()
        if fixed_trees is not None:
            if fixed_trees <= 0:
                raise ValueError("fixed_trees must be positive")
            base = xgb.train(
                fit_params, self.dm_train,
                evals=[(self.dm_train, "train"), (self.dm_validation, "validation")],
                num_boost_round=fixed_trees, verbose_eval=False)
            n_trees = int(fixed_trees)
            base_best_rmse = None
        else:
            base_evals = {}
            base = xgb.train(
                fit_params, self.dm_train,
                evals=[(self.dm_train, "train"), (self.dm_validation, "validation")],
                evals_result=base_evals, num_boost_round=MAX_BOOST_ROUNDS,
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=EARLY_STOPPING_ROUNDS, metric_name="rmse",
                    data_name="validation", save_best=True)],
                verbose_eval=False)
            n_trees = int(base.best_iteration) + 1
            base_best_rmse = float(base_evals["validation"]["rmse"][int(base.best_iteration)])
        base_train = self.physical_predict(base, self.dm_train, n_trees)
        base_validation = self.physical_predict(base, self.dm_validation, n_trees)
        base_dev = self.physical_predict(base, self.dm_dev, n_trees)
        train_weight, validation_weight, weight_meta = positive_response_weights(
            base_train, base_validation, alpha, WEIGHT_CAP)
        weighted_train = xgb.DMatrix(self.x_train, label=self.y_train,
                                     weight=train_weight, feature_names=self.features)
        weighted_validation = xgb.DMatrix(self.x_validation, label=self.y_validation,
                                          weight=validation_weight, feature_names=self.features)
        weighted_evals = {}
        weighted = xgb.train(
            fit_params, weighted_train,
            evals=[(weighted_train, "train"), (weighted_validation, "validation")],
            evals_result=weighted_evals, num_boost_round=n_trees, verbose_eval=False)
        weighted_validation_physical = self.physical_predict(weighted, self.dm_validation, n_trees)
        weighted_dev = self.physical_predict(weighted, self.dm_dev, n_trees)
        metrics = {
            "alpha": float(alpha),
            "n_trees": n_trees,
            "fit_seconds": float(time.time() - started),
            "base": {
                "ordinary_validation_r2": r2(self.y_validation_physical, base_validation),
                "vector": {"development": self.vector.score(base_dev, self.dev_cases)},
                **({"best_validation_rmse_standardized": base_best_rmse}
                   if base_best_rmse is not None else {"fixed_tree_count": n_trees}),
            },
            "weighted": {
                "ordinary_validation_r2": r2(self.y_validation_physical,
                                             weighted_validation_physical),
                "vector": {"development": self.vector.score(weighted_dev, self.dev_cases)},
                "final_weighted_validation_rmse_standardized": float(
                    weighted_evals["validation"]["rmse"][-1]),
            },
            "weight": weight_meta,
        }
        del (base_train, base_validation, base_dev, weighted_validation_physical,
             weighted_dev, train_weight, validation_weight,
             weighted_train, weighted_validation)
        gc.collect()
        if keep_models:
            return metrics, base, weighted
        del base, weighted
        gc.collect()
        return metrics, None, None


def common_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "12")),
        "booster": "gbtree",
        "verbosity": 0,
        "max_bin": 256,
        "seed": SEED,
    }


def suggest_params(trial):
    import optuna
    params = {
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    alpha = trial.suggest_float("response_weight_alpha", 0.01, 0.20, log=True)
    return params, float(alpha)


def completed_trials(study):
    import optuna
    return sum(trial.state == optuna.trial.TrialState.COMPLETE
               for trial in study.get_trials(deepcopy=False))


def load_source_task(cfg, source_tag):
    """Frozen source metadata defines the standardization, features, cuts, baseline."""
    model_dir = cfg["training"]["model_dir"]
    path = os.path.join(model_dir, f"emulator_metadata_{source_tag}.json")
    with open(path, encoding="utf-8") as handle:
        task = json.load(handle)["tasks"]["regression"]
    tr = cfg["training"]
    if list(task["features"]) != list(tr["features"]):
        raise RuntimeError("configuration features differ from source metadata")
    if [list(map(float, row)) for row in task["cuts"]] != \
            [list(map(float, row)) for row in tr["regression_cuts"]]:
        raise RuntimeError("configuration cuts differ from source metadata")
    standardization = task["standardization"]
    baseline = {name: task["params"][name] for name in TUNED_PARAM_NAMES
                if name in task["params"]}
    return {
        "metadata_path": path,
        "metadata_sha256": sha256(path),
        "target_mean": float(standardization["mean"]),
        "target_scale": float(standardization["std"]),
        "baseline_params": baseline,
    }


def training_payload(experiment, extra):
    payload = {
        "fit_cases": list(experiment.fit_cases),
        "n_train_rows": experiment.n_train,
        "n_validation_rows": experiment.n_validation,
        "base_loss": "ordinary squared error with RMSE early stopping",
        "weighted_loss": "squared error under positive-response sample weights",
        "weighted_refit_starts_from_scratch": True,
        "weighted_refit_uses_base_selected_tree_count": True,
        "weight_formula": (
            "(1 + alpha * min(max(base_prediction,0)^2 / "
            "train_mean_power, 50)) / train_mean_raw_weight"
        ),
        "weight_uses_label": False,
        "both_stages_retrained_for_every_trial": True,
        "features": experiment.features,
        "target_standardization": {
            "mean": experiment.target_mean, "std": experiment.target_scale,
        },
    }
    payload.update(extra)
    return payload


def run_tune(args, cfg, source, population):
    import optuna
    experiment = ReweightedExperiment(
        population, args.fit_cases, args.dev_cases,
        source["target_mean"], source["target_scale"],
        cfg["training"]["test_size"], cfg["training"]["random_state"])
    output = args.output_dir
    summary_path = os.path.join(output, "summary.json")
    if os.path.exists(summary_path):
        print(f"Study already finalized: {summary_path}", flush=True)
        return
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10, multivariate=True)
    database = os.path.join(output, "study.sqlite3")
    study = optuna.create_study(
        study_name=f"reweighted_vector_{args.source_tag}",
        storage=f"sqlite:///{database}", direction="minimize",
        sampler=sampler, load_if_exists=True)
    if len(study.trials) == 0:
        baseline = source["baseline_params"]
        for alpha in ENQUEUE_ALPHAS:
            study.enqueue_trial({**baseline, "response_weight_alpha": alpha})

    def objective(trial):
        params, alpha = suggest_params(trial)
        metrics, _, _ = experiment.fit_candidate(params, alpha, keep_models=False)
        for key, value in metrics.items():
            trial.set_user_attr(key, json_clean(value))
        slope = metrics["weighted"]["vector"]["development"]["slope_measured_on_predicted"]["mean"]
        score = abs(float(slope) - 1.0)
        base_slope = metrics["base"]["vector"]["development"]["slope_measured_on_predicted"]["mean"]
        print(f"TRIAL_DONE trial={trial.number} score={score:.8f} alpha={alpha:.6f} "
              f"trees={metrics['n_trees']} base_dev_slope={base_slope:.6f} "
              f"weighted_dev_slope={slope:.6f} "
              f"pair_r2={metrics['weighted']['ordinary_validation_r2']:.8f} "
              f"seconds={metrics['fit_seconds']:.1f}", flush=True)
        return score

    remaining = args.n_trials - completed_trials(study)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                       show_progress_bar=False)
    if completed_trials(study) < args.n_trials:
        raise RuntimeError(f"only {completed_trials(study)} completed of {args.n_trials} trials")
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(os.path.join(output, "trials.csv"), index=False)

    best = study.best_trial
    best_params = {key: value for key, value in best.params.items()
                   if key != "response_weight_alpha"}
    best_alpha = float(best.params["response_weight_alpha"])
    best_metrics, base_model, weighted_model = experiment.fit_candidate(
        best_params, best_alpha, keep_models=True)
    if base_model is None or weighted_model is None:
        raise RuntimeError("best model refit did not return boosters")
    base_path = os.path.join(output, "best_base_model.json")
    weighted_path = os.path.join(output, "best_weighted_model.json")
    atomic_model(base_model, base_path)
    atomic_model(weighted_model, weighted_path)
    payload = {
        "schema_version": 1,
        "kind": "fresh ordinary base followed by positive-response weighted refit (tuned)",
        "study": {
            "name": study.study_name,
            "database": database,
            "requested_completed_trials": args.n_trials,
            "completed_trials": completed_trials(study),
            "best_trial": best.number,
            "best_value_abs_development_slope_minus_one": best.value,
            "sampler": "Optuna TPESampler",
            "seed": SEED,
        },
        "objective": {
            "metric": "abs(case-balanced half-shear vector slope - 1)",
            "selection_cases": list(args.dev_cases),
            "ordinary_pair_loss_in_objective": False,
            "coherent_anchor_truth_used": False,
            "constgold_used": False,
        },
        "best_params": best_params,
        "best_alpha": best_alpha,
        "best_metrics": best_metrics,
        "artifacts": {
            "base_model": base_path,
            "base_model_sha256": sha256(base_path),
            "weighted_model": weighted_path,
            "weighted_model_sha256": sha256(weighted_path),
        },
        "source": {
            "tag": args.source_tag,
            "metadata": source["metadata_path"],
            "metadata_sha256": source["metadata_sha256"],
            "catalogue": os.path.join(cfg["simulation"]["output_path"],
                                      "response_catalogue_train.feather"),
        },
    }
    payload["training"] = training_payload(experiment, {})
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(f"REWEIGHTED_EXTNBR_TUNE_DONE trials={completed_trials(study)} best={best.number}",
          flush=True)


def run_train(args, cfg, source, population):
    experiment = ReweightedExperiment(
        population, args.fit_cases, args.dev_cases,
        source["target_mean"], source["target_scale"],
        cfg["training"]["test_size"], cfg["training"]["random_state"])
    output = args.output_dir
    recipe_source = None
    if args.recipe is not None:
        with open(args.recipe, encoding="utf-8") as handle:
            recipe = json.load(handle)
        params = {k: v for k, v in recipe["best_params"].items()}
        alpha = float(recipe["best_alpha"])
        n_trees = int(recipe["best_metrics"]["n_trees"])
        recipe_source = {"summary": args.recipe, "summary_sha256": sha256(args.recipe)}
    else:
        missing = [name for name in ("params_json", "alpha", "n_trees")
                   if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--recipe or all of {missing} are required in train mode")
        with open(args.params_json, encoding="utf-8") as handle:
            params = json.load(handle)
        alpha = float(args.alpha)
        n_trees = int(args.n_trees)
        recipe_source = {"params_json": args.params_json}
    unknown = sorted(set(params) - set(TUNED_PARAM_NAMES))
    if unknown:
        raise SystemExit(f"params contain non-tuned keys {unknown}")

    metrics, base_model, weighted_model = experiment.fit_candidate(
        params, alpha, fixed_trees=n_trees, keep_models=True)
    if base_model is None or weighted_model is None:
        raise RuntimeError("fixed-recipe fit did not return boosters")
    base_path = os.path.join(output, "base_model.json")
    weighted_path = os.path.join(output, "weighted_model.json")
    atomic_model(base_model, base_path)
    atomic_model(weighted_model, weighted_path)
    payload = {
        "schema_version": 1,
        "kind": "fixed-recipe ordinary base followed by positive-response weighted refit",
        "recipe": recipe_source,
        "fixed_params": params,
        "alpha": alpha,
        "n_trees": n_trees,
        "metrics": metrics,
        "artifacts": {
            "base_model": base_path,
            "base_model_sha256": sha256(base_path),
            "weighted_model": weighted_path,
            "weighted_model_sha256": sha256(weighted_path),
        },
        "source": {
            "tag": args.source_tag,
            "metadata": source["metadata_path"],
            "metadata_sha256": source["metadata_sha256"],
            "catalogue": os.path.join(cfg["simulation"]["output_path"],
                                      "response_catalogue_train.feather"),
        },
    }
    payload["training"] = training_payload(
        experiment, {"base_loss": "ordinary squared error at the fixed tree count"})
    atomic_json(os.path.join(output, "summary.json"), payload)

    if args.model_tag:
        # BlendEMU deployment convention: write the sidecar BlendingPredictor.load reads.
        tr = cfg["training"]
        tr["model_tag"] = args.model_tag
        model_path = TE._fname(tr["model_dir"], "regression_model.json", cfg)
        curve_path = TE._fname(tr["model_dir"], "regression_train_curve.npz", cfg)
        if os.path.exists(model_path):
            raise FileExistsError(f"refusing to overwrite {model_path}")
        weighted_model.save_model(model_path)
        np.savez(curve_path, weighted_rmse=np.asarray(
            [metrics["weighted"]["final_weighted_validation_rmse_standardized"]], float))
        boundary = np.array([[float(lo), float(hi)] for lo, hi in zip(
            experiment.x_train.min(axis=0), experiment.x_train.max(axis=0))])
        fit_params = common_params()
        fit_params.update(params)
        TE._update_metadata(
            cfg, "regression", model_path, experiment.features, boundary, fit_params,
            standardization=(experiment.target_mean, experiment.target_scale),
            train_curve_path=curve_path,
            metrics={
                "best_iteration": n_trees - 1,
                "best_trees": n_trees,
                "score_name": "weighted_rmse",
                "response_power_weight": {
                    "formula": "(1 + alpha * min(max(base_prediction,0)^2 / "
                               "train_mean_power, cap)) / train_mean_raw_weight",
                    "alpha": alpha,
                    "cap": WEIGHT_CAP,
                    "label_used_in_weight": False,
                    "fixed_tree_count": n_trees,
                    "fit_cases": list(experiment.fit_cases),
                },
            })
        print(f"saved sidecar model={model_path}")
    print("REWEIGHTED_EXTNBR_TRAIN_DONE", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["tune", "train"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-tag", default=None,
                        help="frozen emulator tag defining standardization/features/cuts "
                             "(default: training.model_tag)")
    parser.add_argument("--fit-cases", nargs=2, type=int, default=[40, 199], metavar=("MIN", "MAX"))
    parser.add_argument("--dev-cases", nargs=2, type=int, default=[0, 39], metavar=("MIN", "MAX"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--recipe", default=None,
                        help="train mode: summary.json of a tune run to replay")
    parser.add_argument("--params-json", default=None,
                        help="train mode without --recipe: JSON of the eight tuned params")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--n-trees", type=int, default=None)
    parser.add_argument("--model-tag", default=None,
                        help="train mode: also write emulator_metadata_<tag>.json + "
                             "regression_model_<tag>.json into training.model_dir")
    args = parser.parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("n-trials must be positive")
    fit_cases = tuple(args.fit_cases)
    dev_cases = tuple(args.dev_cases)
    for window in (fit_cases, dev_cases):
        if window[0] > window[1]:
            raise ValueError(f"case window {window} is empty")

    cfg = load_config(args.config)
    source_tag = args.source_tag or cfg["training"]["model_tag"]
    source = load_source_task(cfg, source_tag)
    print(f"### REWEIGHTED EXTNBR mode={args.mode} source_tag={source_tag} "
          f"fit_cases={fit_cases} dev_cases={dev_cases} ###")
    case_min = min(fit_cases[0], dev_cases[0])
    case_max = max(fit_cases[1], dev_cases[1])
    population = load_pair_population(cfg, case_min, case_max)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "tune":
        run_tune(args, cfg, source, population)
    else:
        run_train(args, cfg, source, population)


if __name__ == "__main__":
    main()

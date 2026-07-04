"""
Train or tune XGBoost blending emulators.

Modes:
  tune   — Run Optuna hyperparameter search, then train final model by default
  train  — Train final model using best params (from Optuna study or defaults)

Usage:
  # Tune hyperparameters (GPU recommended)
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task regression
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task regression --tune-only
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task both --n-trials 200

  # Train with best params from a previous Optuna study
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task both

  # Train with built-in default params (no Optuna study needed)
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task regression --no-optuna
"""

import argparse
import fcntl
import json
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from blendemu import data_utils
from blendemu.config import load_config, config_summary, resolve_simulation_set, simulation_shear_scale


def _fname(model_dir, base, cfg):
    """Build a filename that includes the model tag (e.g. 'regression_model_lsst_r.json')."""
    tag = cfg['training'].get('model_tag')
    suffix = f'_{tag}' if tag else ''
    name, ext = os.path.splitext(base)
    return os.path.join(model_dir, f'{name}{suffix}{ext}')


def _json_ready(value):
    """Convert NumPy/scalar containers to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _metadata_path(model_dir, cfg):
    return _fname(model_dir, 'emulator_metadata.json', cfg)


def _catalogue_simulation_set(cfg, set_name):
    """Resolve the simulation settings used by a training catalogue."""
    overrides = cfg.get('catalogues', {}).get(set_name, {})
    return resolve_simulation_set(cfg['simulation'], set_name, overrides=overrides)


def _catalogue_shear_scale(cfg, set_name):
    sim_set = _catalogue_simulation_set(cfg, set_name)
    return simulation_shear_scale(sim_set, label=f'simulation.{set_name}')


def _early_stopping_callback(cfg, task, metric_name, maximize=None):
    """Build a fresh XGBoost early-stopping callback for one training run."""
    tr = cfg['training']
    rounds = int(tr.get(f'{task}_early_stopping_rounds', tr.get('early_stopping_rounds', 30)))
    min_delta = float(
        tr.get(
            f'{task}_early_stopping_min_delta',
            tr.get('early_stopping_min_delta', 0.0),
        )
    )
    kwargs = {
        'rounds': rounds,
        'metric_name': metric_name,
        'data_name': 'eval',
        'maximize': maximize,
        'save_best': True,
        'min_delta': min_delta,
    }
    try:
        return xgb.callback.EarlyStopping(**kwargs)
    except TypeError:
        # Older XGBoost releases do not support min_delta.
        kwargs.pop('min_delta')
        return xgb.callback.EarlyStopping(**kwargs)


def _metric_at_best(evals_result, model, data_name, metric_name):
    """Return the metric value at XGBoost's selected best iteration."""
    values = evals_result[data_name][metric_name]
    best_iteration = getattr(model, 'best_iteration', None)
    if best_iteration is None:
        return values[-1]
    return values[min(int(best_iteration), len(values) - 1)]


def _update_metadata(cfg, task, model_path, features, boundary, params,
                     standardization=None, train_curve_path=None, metrics=None):
    """Update the JSON sidecar that keeps model metadata in one place."""
    model_dir = cfg['training']['model_dir']
    path = _metadata_path(model_dir, cfg)
    os.makedirs(model_dir, exist_ok=True)

    lock_path = f'{path}.lock'
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if os.path.exists(path):
            with open(path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        tr = cfg['training']
        metadata.update({
            'format_version': 1,
            'tag': tr.get('model_tag'),
            'training': _json_ready({
                'rescale': tr.get('rescale'),
                'test_size': tr.get('test_size'),
                'random_state': tr.get('random_state'),
            }),
        })

        task_meta = {
            'model_file': os.path.basename(model_path),
            'features': _json_ready(list(features)),
            'boundary': _json_ready(boundary),
            'params': _json_ready(params),
        }
        if standardization is not None:
            y_mean, y_std = standardization
            task_meta['standardization'] = {
                'mean': _json_ready(float(y_mean)),
                'std': _json_ready(float(y_std)),
            }
        if train_curve_path is not None:
            task_meta['train_curve_file'] = os.path.basename(train_curve_path)
        if metrics is not None:
            task_meta['metrics'] = _json_ready(metrics)

        cut_key = {
            'classification': 'classification_cuts',
            'regression': 'regression_cuts',
            'self_response': 'self_response_cuts',
        }.get(task)
        if cut_key is not None:
            task_meta['cuts'] = _json_ready(tr.get(cut_key))

        # Persist the neighbour aperture (r_max in arcsec, k) the pair catalogue
        # was built with, so inference matches training. Map task -> catalogue set.
        cat_set = {
            'classification': 'detection',
            'regression': 'response',
            'self_response': 'self_response',
        }.get(task)
        cat_cfg = cfg.get('catalogues', {}).get(cat_set, {}) if cat_set else {}
        if 'r_max' in cat_cfg:
            task_meta['r_max'] = _json_ready(cat_cfg['r_max'])
        if 'k' in cat_cfg:
            task_meta['k'] = _json_ready(cat_cfg['k'])

        metadata.setdefault('tasks', {})[task] = task_meta

        tmp_path = f'{path}.{os.getpid()}.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        os.replace(tmp_path, path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return path


# Fallback params if no Optuna study exists. Tuned on galsbi/smf50 r-band
# simulations. Prefer running --mode tune first to produce a survey-specific
# study, then --mode train to use its best params.
DEFAULT_PARAMS = {
    'regression': {
        'subsample': 0.365,
        'colsample_bytree': 0.979,
        'learning_rate': 0.00313,
        'max_depth': 6,
        'min_child_weight': 189,
    },
    'classification': {
        'subsample': 0.912,
        'colsample_bytree': 0.864,
        'learning_rate': 0.191,
        'max_depth': 10,
        'min_child_weight': 4,
        'reg_lambda': 0.298,
        'reg_alpha': 0.361,
        'gamma': 0.342,
    },
    # self_response: no tuned defaults — must run --mode tune first or pass
    # explicit params. Falling back would silently use blending params, which
    # fit a very different target distribution.
}


# ──────────────────────────────────────────────
# Data loading (shared between tune and train)
# ──────────────────────────────────────────────

def load_regression_data(cfg):
    """Load and preprocess response catalogue for regression."""
    tr = cfg['training']
    out = cfg['simulation']['output_path']
    rc = tr['rescale']

    cat_path = os.path.join(out, 'response_catalogue_train.feather')
    print(f"Loading: {cat_path}")

    dataset = pd.read_feather(cat_path)
    print(f"  Raw: {dataset.shape[0]:,} rows")

    dataset = data_utils.source_select_reg(dataset, cuts=tr['regression_cuts'])
    print(f"  After cuts: {dataset.shape[0]:,} rows")

    dataset = data_utils.rescale(dataset, pixel_rms=rc['pixel_rms'],
                                 pixel_size=rc['pixel_size'], zero_mag=rc['zero_mag'],
                                 psf_fwhm=rc['psf_fwhm'], moffat_beta=rc['moffat_beta'])

    shear = _catalogue_shear_scale(cfg, 'response')
    target = 'delta_et1'
    y = dataset[target] / shear

    y_mean, y_std = float(np.mean(y)), float(np.std(y, ddof=1))
    y = data_utils.standardize(y, y_mean, y_std)
    print(f"  Standardized: mean={y_mean:.6f}, std={y_std:.4f}")

    features = tr['features']
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[features], y, test_size=tr['test_size'], random_state=tr['random_state'],
    )
    print(f"  Train: {x_train.shape[0]:,}, Test: {x_test.shape[0]:,}")

    return (xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_test, y_test),
            x_train, x_test, y_train, y_test, y_mean, y_std)


def load_self_response_data(cfg):
    """Load and preprocess self-response catalogue.

    Same shape as regression data but reads from self_response_catalogue_train.feather
    and uses self_response_cuts instead of regression_cuts.
    """
    tr = cfg['training']
    out = cfg['simulation']['output_path']
    rc = tr['rescale']

    sr_cfg = cfg.get('catalogues', {}).get('self_response', {})
    output_prefix = sr_cfg.get('output_prefix', 'self_response_catalogue')
    cat_path = os.path.join(out, f'{output_prefix}_train.feather')
    print(f"Loading: {cat_path}")

    dataset = pd.read_feather(cat_path)
    print(f"  Raw: {dataset.shape[0]:,} rows")

    # Self-response uses looser cuts (target galaxies span a wider distribution
    # than the primaries used for blending response).
    cuts = tr.get('self_response_cuts', tr['regression_cuts'])
    dataset = data_utils.source_select_self_response(dataset, cuts=cuts)
    print(f"  After cuts: {dataset.shape[0]:,} rows")

    dataset = data_utils.rescale(dataset, pixel_rms=rc['pixel_rms'],
                                 pixel_size=rc['pixel_size'], zero_mag=rc['zero_mag'],
                                 psf_fwhm=rc['psf_fwhm'], moffat_beta=rc['moffat_beta'])

    shear = _catalogue_shear_scale(cfg, 'self_response')
    target = 'delta_et1'
    y = dataset[target] / shear
    print(f"  Before standardize: mean(delta_et1/gamma)={y.mean():.4f} "
          "(high-quality isolated galaxies should approach ~1)")

    y_mean, y_std = float(np.mean(y)), float(np.std(y, ddof=1))
    y = data_utils.standardize(y, y_mean, y_std)
    print(f"  Standardized: mean={y_mean:.6f}, std={y_std:.4f}")

    features = tr['features']
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[features], y, test_size=tr['test_size'], random_state=tr['random_state'],
    )
    print(f"  Train: {x_train.shape[0]:,}, Test: {x_test.shape[0]:,}")

    return (xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_test, y_test),
            x_train, x_test, y_train, y_test, y_mean, y_std)


def load_classification_data(cfg):
    """Load and preprocess detection catalogue for classification."""
    tr = cfg['training']
    out = cfg['simulation']['output_path']
    rc = tr['rescale']

    cat_path = os.path.join(out, 'detection_catalogue_train.feather')
    print(f"Loading: {cat_path}")

    dataset = pd.read_feather(cat_path)
    print(f"  Raw: {dataset.shape[0]:,} rows")

    dataset = data_utils.source_select_cla(dataset, cuts=tr['classification_cuts'])
    print(f"  After cuts: {dataset.shape[0]:,} rows")

    dataset = data_utils.rescale(dataset, pixel_rms=rc['pixel_rms'],
                                 pixel_size=rc['pixel_size'], zero_mag=rc['zero_mag'],
                                 psf_fwhm=rc['psf_fwhm'], moffat_beta=rc['moffat_beta'])

    features = tr.get('classification_features', tr['features'])
    y = dataset['detected']
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[features], y, test_size=tr['test_size'], random_state=tr['random_state'],
    )
    print(f"  Train: {x_train.shape[0]:,}, Test: {x_test.shape[0]:,}")
    print(f"  Positive rate: {y_train.mean():.3f}")

    return (xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_test, y_test),
            x_train, x_test, y_train, y_test)


# ──────────────────────────────────────────────
# Tuning
# ──────────────────────────────────────────────

def _study_name(task, cfg):
    """Optuna study name, namespaced by model tag when one is configured."""
    tag = cfg['training'].get('model_tag')
    if not tag:
        return task
    safe_tag = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(tag))
    return f'{task}_{safe_tag}'


def _study_db_path(task, cfg):
    study_dir = os.path.join(cfg['training']['model_dir'], 'studies')
    os.makedirs(study_dir, exist_ok=True)
    return os.path.join(study_dir, f'{_study_name(task, cfg)}.db')


def _study_storage(task, cfg):
    """Optuna storage with a longer SQLite timeout for parallel workers."""
    db_path = _study_db_path(task, cfg)
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={
            'connect_args': {'timeout': 300},
            'pool_pre_ping': True,
        },
    )


def _load_or_create_study(task, cfg, direction):
    """Create or load a study, serializing first-time SQLite initialization."""
    db_path = _study_db_path(task, cfg)
    study_name = _study_name(task, cfg)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=20,
        constant_liar=True,
    )
    lock_path = f'{db_path}.init.lock'
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        study = optuna.create_study(
            study_name=study_name,
            storage=_study_storage(task, cfg),
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
        )
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return study


def _save_sampler(task, cfg, sampler):
    study_dir = os.path.dirname(_study_db_path(task, cfg))
    study_name = _study_name(task, cfg)
    sampler_path = os.path.join(study_dir, f'{study_name}_sampler_{os.getpid()}.pkl')
    with open(sampler_path, 'wb') as f:
        pickle.dump(sampler, f)


def tune_regression(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for regression."""
    def objective(trial):
        param = {
            "n_jobs": -1, "device": "cuda", "booster": "gbtree",
            "verbosity": 0, "objective": "reg:squarederror",
            "disable_default_eval_metric": 1,
            "subsample": trial.suggest_float("subsample", 0.3, 1.),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10., log=True),
            "gamma": trial.suggest_float("gamma", 0., 5.),
        }
        custom_metric = lambda predt, dtrain: ("r2", r2_score(dtrain.get_label(), predt))
        evals_result = {}
        bst = xgb.train(param, DMtrain,
                        evals=[(DMtrain, "train"), (DMtest, "eval")],
                        evals_result=evals_result,
                        num_boost_round=1000, verbose_eval=False,
                        custom_metric=custom_metric, maximize=True,
                        callbacks=[_early_stopping_callback(cfg, 'regression', 'r2', maximize=True)])
        ev = _metric_at_best(evals_result, bst, 'eval', 'r2')
        tr = _metric_at_best(evals_result, bst, 'train', 'r2')
        return ev - 0.6 * abs(tr - ev)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = _study_name('regression', cfg)
    study = _load_or_create_study('regression', cfg, direction='maximize')
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    _save_sampler('regression', cfg, study.sampler)
    return study


def tune_self_response(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for self-response (same objective as regression)."""
    def objective(trial):
        param = {
            "n_jobs": -1, "device": "cuda", "booster": "gbtree",
            "verbosity": 0, "objective": "reg:squarederror",
            "disable_default_eval_metric": 1,
            "subsample": trial.suggest_float("subsample", 0.3, 1.),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10., log=True),
            "gamma": trial.suggest_float("gamma", 0., 5.),
        }
        custom_metric = lambda predt, dtrain: ("r2", r2_score(dtrain.get_label(), predt))
        evals_result = {}
        bst = xgb.train(param, DMtrain,
                        evals=[(DMtrain, "train"), (DMtest, "eval")],
                        evals_result=evals_result,
                        num_boost_round=1000, verbose_eval=False,
                        custom_metric=custom_metric, maximize=True,
                        callbacks=[_early_stopping_callback(cfg, 'self_response', 'r2', maximize=True)])
        ev = _metric_at_best(evals_result, bst, 'eval', 'r2')
        tr = _metric_at_best(evals_result, bst, 'train', 'r2')
        return ev - 0.6 * abs(tr - ev)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = _study_name('self_response', cfg)
    study = _load_or_create_study('self_response', cfg, direction='maximize')
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    _save_sampler('self_response', cfg, study.sampler)
    return study


def tune_classification(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for classification."""
    def objective(trial):
        param = {
            "n_jobs": -1, "device": "cuda", "booster": "gbtree",
            "verbosity": 0, "objective": "binary:logistic",
            "subsample": trial.suggest_float("subsample", 0.3, 1.),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10., log=True),
            "gamma": trial.suggest_float("gamma", 0., 5.),
        }
        evals_result = {}
        bst = xgb.train(param, DMtrain,
                        evals=[(DMtrain, "train"), (DMtest, "eval")],
                        evals_result=evals_result,
                        num_boost_round=1000, verbose_eval=False,
                        callbacks=[_early_stopping_callback(
                            cfg, 'classification', 'logloss', maximize=False)])
        ev = _metric_at_best(evals_result, bst, 'eval', 'logloss')
        tr = _metric_at_best(evals_result, bst, 'train', 'logloss')
        return ev + 0.6 * abs(tr - ev)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = _study_name('classification', cfg)
    study = _load_or_create_study('classification', cfg, direction='minimize')
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    _save_sampler('classification', cfg, study.sampler)
    return study


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def _get_best_params(task, cfg, use_optuna):
    """
    Load best params from Optuna study, or fall back to built-in defaults.

    Raises
    ------
    RuntimeError
        If no study exists for `task` and no built-in defaults are available
        (e.g. 'self_response' has no defaults by design).
    """
    if use_optuna:
        study_name = _study_name(task, cfg)
        db_path = _study_db_path(task, cfg)
        if os.path.exists(db_path):
            study = optuna.load_study(study_name=study_name, storage=_study_storage(task, cfg))
            print(f"  Loaded Optuna study: {len(study.trials)} trials, best={study.best_value:.6f}")
            return study.best_params
        print(f"  No Optuna study at {db_path}, falling back to DEFAULT_PARAMS")
    if task not in DEFAULT_PARAMS:
        raise RuntimeError(
            f"No DEFAULT_PARAMS for task={task!r}. Run `--mode tune --task {task}` "
            "first to produce an Optuna study with best params.")
    return DEFAULT_PARAMS[task]


def train_regression(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna):
    """Train regression model with best params."""
    model_dir = cfg['training']['model_dir']
    features = cfg['training']['features']

    print("\n--- Training regression model ---")
    params = dict(_get_best_params('regression', cfg, use_optuna))
    params.update({
        'objective': 'reg:squarederror', 'n_jobs': -1, 'device': 'cuda',
        'booster': 'gbtree', 'disable_default_eval_metric': 1,
    })
    print(f"  Params: {params}")

    custom_metric = lambda predt, dtrain: ("r2", r2_score(dtrain.get_label(), predt))

    t0 = time.time()
    evals_result = {}
    bst = xgb.train(params, DMtrain,
                    evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=evals_result,
                    num_boost_round=2000, verbose_eval=100,
                    custom_metric=custom_metric, maximize=True,
                    callbacks=[_early_stopping_callback(cfg, 'regression', 'r2', maximize=True)])

    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  Time: {time.time()-t0:.1f}s, best_iter={bst.best_iteration}, trees={best_trees}, R²: {bst.best_score:.6f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = _fname(model_dir, 'regression_model.json', cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    train_curve_path = _fname(model_dir, 'regression_train_curve.npz', cfg)
    bst.save_model(model_path)
    np.savez(train_curve_path,
             train_r2=evals_result['train']['r2'],
             eval_r2=evals_result['eval']['r2'])
    metadata_path = _update_metadata(
        cfg, 'regression', model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=train_curve_path,
        metrics={
            'best_iteration': bst.best_iteration,
            'best_trees': best_trees,
            'best_score': bst.best_score,
            'score_name': 'r2',
        },
    )

    print(f"  Saved to {model_dir}")
    print(f"  Metadata: {metadata_path}")
    return bst


def train_self_response(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna):
    """Train self-response regression model with best params."""
    model_dir = cfg['training']['model_dir']
    features = cfg['training']['features']

    print("\n--- Training self-response model ---")
    params = dict(_get_best_params('self_response', cfg, use_optuna))
    params.update({
        'objective': 'reg:squarederror', 'n_jobs': -1, 'device': 'cuda',
        'booster': 'gbtree', 'disable_default_eval_metric': 1,
    })
    print(f"  Params: {params}")

    custom_metric = lambda predt, dtrain: ("r2", r2_score(dtrain.get_label(), predt))

    t0 = time.time()
    evals_result = {}
    bst = xgb.train(params, DMtrain,
                    evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=evals_result,
                    num_boost_round=2000, verbose_eval=100,
                    custom_metric=custom_metric, maximize=True,
                    callbacks=[_early_stopping_callback(cfg, 'self_response', 'r2', maximize=True)])

    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  Time: {time.time()-t0:.1f}s, best_iter={bst.best_iteration}, trees={best_trees}, R²: {bst.best_score:.6f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = _fname(model_dir, 'self_response_model.json', cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    train_curve_path = _fname(model_dir, 'self_response_train_curve.npz', cfg)
    bst.save_model(model_path)
    np.savez(train_curve_path,
             train_r2=evals_result['train']['r2'],
             eval_r2=evals_result['eval']['r2'])
    metadata_path = _update_metadata(
        cfg, 'self_response', model_path, features, boundary, params,
        standardization=(y_mean, y_std), train_curve_path=train_curve_path,
        metrics={
            'best_iteration': bst.best_iteration,
            'best_trees': best_trees,
            'best_score': bst.best_score,
            'score_name': 'r2',
        },
    )

    print(f"  Saved to {model_dir}")
    print(f"  Metadata: {metadata_path}")
    return bst


def train_classification(cfg, DMtrain, DMtest, x_train, y_test, use_optuna):
    """Train classification model with best params."""
    model_dir = cfg['training']['model_dir']
    features = list(x_train.columns)

    print("\n--- Training classification model ---")
    params = dict(_get_best_params('classification', cfg, use_optuna))
    params.update({
        'objective': 'binary:logistic', 'n_jobs': -1,
        'device': 'cuda', 'booster': 'gbtree',
    })
    print(f"  Params: {params}")

    t0 = time.time()
    evals_result = {}
    bst = xgb.train(params, DMtrain,
                    evals=[(DMtrain, "train"), (DMtest, "eval")],
                    evals_result=evals_result,
                    num_boost_round=2000, verbose_eval=50,
                    callbacks=[_early_stopping_callback(
                        cfg, 'classification', 'logloss', maximize=False)])

    best_trees = data_utils.get_xgb_iteration_range(bst)[1]
    print(f"  Time: {time.time()-t0:.1f}s, best_iter={bst.best_iteration}, trees={best_trees}, logloss: {bst.best_score:.6f}")

    y_pred = bst.predict(DMtest, iteration_range=data_utils.get_xgb_iteration_range(bst))
    print(f"  Accuracy: {accuracy_score(y_test, np.round(y_pred)):.4f}")
    print(f"  Balanced: {balanced_accuracy_score(y_test, np.round(y_pred)):.4f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = _fname(model_dir, 'classification_model.json', cfg)
    boundary = np.array([[x_train[f].min(), x_train[f].max()] for f in features])
    train_curve_path = _fname(model_dir, 'classification_train_curve.npz', cfg)
    bst.save_model(model_path)
    np.savez(train_curve_path,
             train_logloss=evals_result['train']['logloss'],
             eval_logloss=evals_result['eval']['logloss'])
    metadata_path = _update_metadata(
        cfg, 'classification', model_path, features, boundary, params,
        train_curve_path=train_curve_path,
        metrics={
            'best_iteration': bst.best_iteration,
            'best_trees': best_trees,
            'best_score': bst.best_score,
            'score_name': 'logloss',
            'accuracy': accuracy_score(y_test, np.round(y_pred)),
            'balanced_accuracy': balanced_accuracy_score(y_test, np.round(y_pred)),
        },
    )

    print(f"  Saved to {model_dir}")
    print(f"  Metadata: {metadata_path}")
    return bst


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tune or train blendemu emulators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune regression with 200 Optuna trials
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task regression

  # Train both models using best params from Optuna study
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task both

  # Train using built-in default params (skip Optuna)
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task both --no-optuna
        """,
    )
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--mode', type=str, required=True, choices=['tune', 'train'],
                        help='tune = Optuna search, train = final model')
    parser.add_argument('--task', type=str, default='both',
                        choices=['regression', 'self_response', 'classification', 'both', 'all'])
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Optuna trials (default: from config)')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Use default params instead of loading from Optuna study')
    parser.add_argument('--tune-only', action='store_true',
                        help='With --mode tune, run only Optuna trials and skip final model training')

    args = parser.parse_args()
    if args.tune_only and args.mode != 'tune':
        parser.error('--tune-only is only valid with --mode tune')

    cfg = load_config(args.config)
    print(config_summary(cfg))
    print()

    n_trials = args.n_trials or cfg['training']['n_trials']
    use_optuna = not args.no_optuna
    train_after_tune = args.mode == 'tune' and not args.tune_only

    t_total = time.time()

    if args.task in ('regression', 'both', 'all'):
        print("=" * 60)
        print(f"REGRESSION ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = load_regression_data(cfg)

        if args.mode == 'tune':
            tune_regression(cfg, DMtrain, DMtest, n_trials)
        if args.mode == 'train' or train_after_tune:
            train_regression(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna)

        del DMtrain, DMtest, x_train, x_test, y_train, y_test

    if args.task in ('self_response', 'all'):
        print("\n" + "=" * 60)
        print(f"SELF-RESPONSE ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = load_self_response_data(cfg)

        if args.mode == 'tune':
            tune_self_response(cfg, DMtrain, DMtest, n_trials)
        if args.mode == 'train' or train_after_tune:
            train_self_response(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna)

        del DMtrain, DMtest, x_train, x_test, y_train, y_test

    if args.task in ('classification', 'both', 'all'):
        print("\n" + "=" * 60)
        print(f"CLASSIFICATION ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test = load_classification_data(cfg)

        if args.mode == 'tune':
            tune_classification(cfg, DMtrain, DMtest, n_trials)
        if args.mode == 'train' or train_after_tune:
            train_classification(cfg, DMtrain, DMtest, x_train, y_test, use_optuna)

    print(f"\nTotal time: {time.time()-t_total:.0f}s")


if __name__ == '__main__':
    main()

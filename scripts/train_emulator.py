"""
Train or tune XGBoost blending emulators.

Modes:
  tune   — Run Optuna hyperparameter search, save study + best params
  train  — Train final model using best params (from Optuna study or defaults)

Usage:
  # Tune hyperparameters (GPU recommended)
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task regression
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode tune --task both --n-trials 200

  # Train with best params from a previous Optuna study
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task both

  # Train with built-in default params (no Optuna study needed)
  python train_emulator.py --config ../configs/fs2_lsst_r.yaml --mode train --task regression --no-optuna
"""

import argparse
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
from blendemu.config import load_config, config_summary


def _fname(model_dir, base, cfg):
    """Build a filename that includes the model tag (e.g. 'regression_model_lsst_r.json')."""
    tag = cfg['training'].get('model_tag')
    suffix = f'_{tag}' if tag else ''
    name, ext = os.path.splitext(base)
    return os.path.join(model_dir, f'{name}{suffix}{ext}')


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

    shear = cfg['simulation']['shear_values'][1]
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

    cat_path = os.path.join(out, 'self_response_catalogue_train.feather')
    print(f"Loading: {cat_path}")

    dataset = pd.read_feather(cat_path)
    print(f"  Raw: {dataset.shape[0]:,} rows")

    # Self-response uses looser cuts (target galaxies span a wider distribution
    # than the primaries used for blending response).
    cuts = tr.get('self_response_cuts', tr['regression_cuts'])
    dataset = data_utils.source_select_reg(dataset, cuts=cuts)
    print(f"  After cuts: {dataset.shape[0]:,} rows")

    dataset = data_utils.rescale(dataset, pixel_rms=rc['pixel_rms'],
                                 pixel_size=rc['pixel_size'], zero_mag=rc['zero_mag'],
                                 psf_fwhm=rc['psf_fwhm'], moffat_beta=rc['moffat_beta'])

    shear = cfg['simulation']['shear_values'][1]
    target = 'delta_et1'
    y = dataset[target] / shear
    print(f"  Before standardize: mean(delta_et1/gamma)={y.mean():.4f} (expect ~1 for self-response)")

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

    features = tr['features']
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

def tune_regression(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for regression."""
    model_dir = cfg['training']['model_dir']
    study_dir = os.path.join(model_dir, 'studies')
    os.makedirs(study_dir, exist_ok=True)

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
        xgb.train(param, DMtrain,
                  evals=[(DMtrain, "train"), (DMtest, "eval")],
                  evals_result=evals_result,
                  num_boost_round=1000, early_stopping_rounds=20, verbose_eval=False,
                  custom_metric=custom_metric, maximize=True)
        ev = evals_result['eval']['r2'][-1]
        tr = evals_result['train']['r2'][-1]
        return ev - 0.6 * abs(tr - ev)

    study_name = 'regression'
    storage = f"sqlite:///{study_dir}/{study_name}.db"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20),
    )
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    with open(os.path.join(study_dir, 'regression_sampler.pkl'), 'wb') as f:
        pickle.dump(study.sampler, f)

    return study


def tune_self_response(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for self-response (same objective as regression)."""
    model_dir = cfg['training']['model_dir']
    study_dir = os.path.join(model_dir, 'studies')
    os.makedirs(study_dir, exist_ok=True)

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
        xgb.train(param, DMtrain,
                  evals=[(DMtrain, "train"), (DMtest, "eval")],
                  evals_result=evals_result,
                  num_boost_round=1000, early_stopping_rounds=20, verbose_eval=False,
                  custom_metric=custom_metric, maximize=True)
        ev = evals_result['eval']['r2'][-1]
        tr = evals_result['train']['r2'][-1]
        return ev - 0.6 * abs(tr - ev)

    study_name = 'self_response'
    storage = f"sqlite:///{study_dir}/{study_name}.db"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20),
    )
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    with open(os.path.join(study_dir, 'self_response_sampler.pkl'), 'wb') as f:
        pickle.dump(study.sampler, f)

    return study


def tune_classification(cfg, DMtrain, DMtest, n_trials):
    """Run Optuna hyperparameter search for classification."""
    model_dir = cfg['training']['model_dir']
    study_dir = os.path.join(model_dir, 'studies')
    os.makedirs(study_dir, exist_ok=True)

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
        xgb.train(param, DMtrain,
                  evals=[(DMtrain, "train"), (DMtest, "eval")],
                  evals_result=evals_result,
                  num_boost_round=1000, early_stopping_rounds=20, verbose_eval=False)
        ev = evals_result['eval']['logloss'][-1]
        tr = evals_result['train']['logloss'][-1]
        return ev + 0.6 * abs(tr - ev)

    study_name = 'classification'
    storage = f"sqlite:///{study_dir}/{study_name}.db"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20),
    )
    print(f"Running {n_trials} Optuna trials (study: {study_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    with open(os.path.join(study_dir, 'classification_sampler.pkl'), 'wb') as f:
        pickle.dump(study.sampler, f)

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
        study_dir = os.path.join(cfg['training']['model_dir'], 'studies')
        db_path = os.path.join(study_dir, f'{task}.db')
        if os.path.exists(db_path):
            study = optuna.load_study(study_name=task, storage=f"sqlite:///{db_path}")
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
    params = _get_best_params('regression', cfg, use_optuna)
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
                    num_boost_round=2000, early_stopping_rounds=30, verbose_eval=100,
                    custom_metric=custom_metric, maximize=True)

    print(f"  Time: {time.time()-t0:.1f}s, iterations: {bst.best_iteration}, R²: {bst.best_score:.6f}")

    os.makedirs(model_dir, exist_ok=True)
    bst.save_model(_fname(model_dir, 'regression_model.json', cfg))
    np.save(_fname(model_dir, 'train_boundary_reg.npy', cfg),
            np.array([[x_train[f].min(), x_train[f].max()] for f in features]))
    np.save(_fname(model_dir, 'train_standardization.npy', cfg), [y_mean, y_std])
    np.savez(_fname(model_dir, 'regression_train_curve.npz', cfg),
             train_r2=evals_result['train']['r2'],
             eval_r2=evals_result['eval']['r2'])

    print(f"  Saved to {model_dir}")
    return bst


def train_self_response(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna):
    """Train self-response regression model with best params."""
    model_dir = cfg['training']['model_dir']
    features = cfg['training']['features']

    print("\n--- Training self-response model ---")
    params = _get_best_params('self_response', cfg, use_optuna)
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
                    num_boost_round=2000, early_stopping_rounds=30, verbose_eval=100,
                    custom_metric=custom_metric, maximize=True)

    print(f"  Time: {time.time()-t0:.1f}s, iterations: {bst.best_iteration}, R²: {bst.best_score:.6f}")

    os.makedirs(model_dir, exist_ok=True)
    bst.save_model(_fname(model_dir, 'self_response_model.json', cfg))
    np.save(_fname(model_dir, 'train_boundary_self.npy', cfg),
            np.array([[x_train[f].min(), x_train[f].max()] for f in features]))
    np.save(_fname(model_dir, 'train_standardization_self.npy', cfg), [y_mean, y_std])
    np.savez(_fname(model_dir, 'self_response_train_curve.npz', cfg),
             train_r2=evals_result['train']['r2'],
             eval_r2=evals_result['eval']['r2'])

    print(f"  Saved to {model_dir}")
    return bst


def train_classification(cfg, DMtrain, DMtest, x_train, y_test, use_optuna):
    """Train classification model with best params."""
    model_dir = cfg['training']['model_dir']
    features = cfg['training']['features']

    print("\n--- Training classification model ---")
    params = _get_best_params('classification', cfg, use_optuna)
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
                    num_boost_round=2000, early_stopping_rounds=30, verbose_eval=50)

    print(f"  Time: {time.time()-t0:.1f}s, iterations: {bst.best_iteration}, logloss: {bst.best_score:.6f}")

    y_pred = bst.predict(DMtest, iteration_range=[0, bst.best_iteration])
    print(f"  Accuracy: {accuracy_score(y_test, np.round(y_pred)):.4f}")
    print(f"  Balanced: {balanced_accuracy_score(y_test, np.round(y_pred)):.4f}")

    os.makedirs(model_dir, exist_ok=True)
    bst.save_model(_fname(model_dir, 'classification_model.json', cfg))
    np.save(_fname(model_dir, 'train_boundary_cla.npy', cfg),
            np.array([[x_train[f].min(), x_train[f].max()] for f in features]))
    np.savez(_fname(model_dir, 'classification_train_curve.npz', cfg),
             train_logloss=evals_result['train']['logloss'],
             eval_logloss=evals_result['eval']['logloss'])

    print(f"  Saved to {model_dir}")
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

    args = parser.parse_args()
    cfg = load_config(args.config)
    print(config_summary(cfg))
    print()

    n_trials = args.n_trials or cfg['training']['n_trials']
    use_optuna = not args.no_optuna

    t_total = time.time()

    if args.task in ('regression', 'both', 'all'):
        print("=" * 60)
        print(f"REGRESSION ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = load_regression_data(cfg)

        if args.mode == 'tune':
            tune_regression(cfg, DMtrain, DMtest, n_trials)
        # Always train after tuning (or just train if mode=train)
        train_regression(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna)

        del DMtrain, DMtest, x_train, x_test, y_train, y_test

    if args.task in ('self_response', 'all'):
        print("\n" + "=" * 60)
        print(f"SELF-RESPONSE ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test, y_mean, y_std = load_self_response_data(cfg)

        if args.mode == 'tune':
            tune_self_response(cfg, DMtrain, DMtest, n_trials)
        train_self_response(cfg, DMtrain, DMtest, x_train, y_mean, y_std, use_optuna)

        del DMtrain, DMtest, x_train, x_test, y_train, y_test

    if args.task in ('classification', 'both', 'all'):
        print("\n" + "=" * 60)
        print(f"CLASSIFICATION ({args.mode})")
        print("=" * 60)
        DMtrain, DMtest, x_train, x_test, y_train, y_test = load_classification_data(cfg)

        if args.mode == 'tune':
            tune_classification(cfg, DMtrain, DMtest, n_trials)
        train_classification(cfg, DMtrain, DMtest, x_train, y_test, use_optuna)

    print(f"\nTotal time: {time.time()-t_total:.0f}s")


if __name__ == '__main__':
    main()

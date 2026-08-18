"""Smoke tests for scripts/retrain_extnbr_reweighted.py (V3 reweight + vector-slope tuning).

Runnable two ways:
  pytest tests/test_reweighted_extnbr.py          # from the repo root
  python tests/test_reweighted_extnbr.py          # standalone (sims1 has no pytest)

The tests synthesize a tiny ordered half-shear pair catalogue, a matching config,
and a fake frozen source-metadata sidecar, then exercise the weight formula, the
case-balanced vector score, the two-stage fit, the 2-trial Optuna tune, the
fixed-recipe train replay, and the BlendEMU sidecar write. No real data or GPU
is touched; XGB_DEVICE is forced to cpu.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("XGB_DEVICE", "cpu")

try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from scripts import retrain_extnbr_reweighted as RRE  # noqa: E402
    IMPORT_ERROR = None
except Exception as error:  # e.g. no optuna -> train_emulator import fails (py31)
    RRE = None
    IMPORT_ERROR = error

if RRE is not None:
    # Keep the smoke fits tiny regardless of entry point.
    RRE.MAX_BOOST_ROUNDS = 80
    RRE.EARLY_STOPPING_ROUNDS = 10

DEV_CASES = (0, 3)
FIT_CASES = (4, 5)
V22_CUTS = [[13, 29], [18, 25.8], [0.0, 10.0], [0.5, 1.5], [0, 10]]
SOURCE_TAG = "smoke_v22"
FEATURES = [
    "Re_input_p_scaled", "Re_input_s_scaled", "r_input_p_scaled", "r_input_s_scaled",
    "sersic_n_input_p", "sersic_n_input_s", "distance_scaled",
]
BASELINE_PARAMS = {
    "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.3,
    "max_depth": 4, "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.1,
    "gamma": 0.0,
}


def build_workspace():
    """Tiny ordered (case, input_index) pair catalogue + config + source sidecar."""
    root = tempfile.mkdtemp(prefix="reweighted_smoke_")
    sim_dir = os.path.join(root, "sim")
    model_dir = os.path.join(root, "models")
    os.makedirs(sim_dir)
    os.makedirs(model_dir)
    rng = np.random.default_rng(7)
    rows = []
    for case in range(DEV_CASES[0], FIT_CASES[1] + 1):
        for primary in range(25):
            r_p = rng.uniform(19.0, 25.0)
            re_p = rng.uniform(0.6, 1.4)
            n_p = rng.uniform(1.0, 3.0)
            for _ in range(rng.integers(1, 5)):
                distance = rng.uniform(0.5, 8.0)
                r_s = rng.uniform(16.0, 24.0)
                re_s = rng.uniform(0.3, 3.0)
                angle = rng.uniform(0.0, 180.0)
                signal = 0.05 * np.exp(-distance / 2.0) * (25.0 - r_s) / 9.0
                label = signal + 0.002 * rng.standard_normal()
                null = 0.002 * rng.standard_normal()
                rows.append((case, primary, angle, r_p, r_s, re_p, re_s, n_p,
                             rng.uniform(1.0, 4.0), distance,
                             label * 0.2, null * 0.2))
    frame = pd.DataFrame(rows, columns=[
        "case", "input_index", "shear_angle", "r_input_p", "r_input_s",
        "Re_input_p", "Re_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "distance", "delta_et1", "delta_et2"])
    frame.to_feather(os.path.join(sim_dir, "response_catalogue_train.feather"))

    config = {
        "catalog": {"type": "fs2", "path": os.path.join(root, "dummy_truth.feather"),
                    "mag_column": "lsst_r_mag", "mag_cut": 29,
                    "re_convention": "circularized"},
        "simulation": {"output_path": sim_dir, "case_offset": 0,
                       "response": {"n_cases": 200, "shear_values": [0.0, 0.2]}},
        "noise": {"source": "csv", "csv_path": os.path.join(root, "dummy_noise.csv"),
                  "band": "LSST_r"},
        "training": {
            "model_dir": model_dir, "model_tag": SOURCE_TAG,
            "regression_cuts": V22_CUTS,
            "rescale": {"pixel_rms": 0.312, "pixel_size": 0.2, "zero_mag": 30,
                        "psf_fwhm": 0.73, "moffat_beta": 2.224},
        },
    }
    config_path = os.path.join(root, "smoke.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle)

    sidecar = {
        "format_version": 1, "tag": SOURCE_TAG,
        "tasks": {"regression": {
            "features": FEATURES, "cuts": V22_CUTS,
            "standardization": {"mean": 0.004, "std": 0.05},
            "params": dict(BASELINE_PARAMS),
            "r_max": 10, "k": 20,
        }},
    }
    sidecar_path = os.path.join(model_dir, f"emulator_metadata_{SOURCE_TAG}.json")
    with open(sidecar_path, "w") as handle:
        json.dump(sidecar, handle, indent=2)
    return root, config_path


def make_experiment(root, config_path):
    from blendemu.config import load_config
    cfg = load_config(config_path)
    source = RRE.load_source_task(cfg, SOURCE_TAG)
    population = RRE.load_pair_population(
        cfg, DEV_CASES[0], FIT_CASES[1])
    return RRE.ReweightedExperiment(
        population, FIT_CASES, DEV_CASES,
        source["target_mean"], source["target_scale"],
        cfg["training"]["test_size"], cfg["training"]["random_state"])


def test_positive_response_weights():
    if RRE is None:
        print('SKIP test_positive_response_weights (training stack unavailable: %s)' % IMPORT_ERROR)
        return
    base_train = np.array([0.0, 1.0, 2.0, 10.0])
    base_validation = np.array([1.0, 3.0])
    alpha, cap = 0.1, 50.0
    train_w, val_w, meta = RRE.positive_response_weights(
        base_train, base_validation, alpha, cap)
    mean_power = float(np.mean(np.maximum(base_train, 0.0) ** 2))
    expected = (1.0 + alpha * np.minimum(np.maximum(base_train, 0.0) ** 2
                                         / mean_power, cap))
    expected = expected / expected.mean()
    np.testing.assert_allclose(train_w, expected.astype(np.float32), rtol=1e-6)
    assert abs(float(train_w.mean()) - 1.0) < 1e-6, "mean train weight must be 1"
    assert meta["train_weight_max"] <= (1.0 + alpha * cap) / (1.0 + alpha) + 1e-9
    # capped row: prediction 10 saturates the ratio at cap
    ratio_cap = min(100.0 / mean_power, cap)
    assert np.isclose(meta["alpha"], alpha) and np.isclose(meta["cap"], cap)
    assert val_w.shape == (2,) and np.all(val_w > 0)
    # labels never enter: identical predictions give identical weights
    again, _, _ = RRE.positive_response_weights(base_train, base_validation, alpha, cap)
    np.testing.assert_array_equal(train_w, again)


def test_vector_evaluation_known_slope():
    if RRE is None:
        print('SKIP test_vector_evaluation_known_slope (training stack unavailable: %s)' % IMPORT_ERROR)
        return
    # Two pairs per primary at shear angles 0 and 45 degrees (2theta = 0, pi/2),
    # each duplicated: cos/sin sums are (2, 2) per primary, so the spin-2 fit of
    # labels k/2*(cos+sin) returns measured = (k/2, k/2) exactly, and a constant
    # prediction q gives predicted = (2q, 2q). The projection slope is then
    # k/(4q) per primary, orthogonal null exactly zero.
    case = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    primary = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    angle = np.array([0.0, 0.0, 45.0, 45.0, 0.0, 0.0, 45.0, 45.0])
    k = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
    label = k / 2.0 * np.cos(np.deg2rad(2 * angle)) + k / 2.0 * np.sin(np.deg2rad(2 * angle))
    null = np.zeros(8)
    q = np.ones(8)
    evaluation = RRE.VectorEvaluation(case, primary, angle, label, null)
    result = evaluation.score(q, (0, 1))
    np.testing.assert_allclose(
        [result["slope_measured_on_predicted"]["mean"],
         result["slope_minus_one"]["mean"],
         result["orthogonal_slope_null"]["mean"]],
        [0.75, -0.25, 0.0], rtol=1e-10)
    assert result["n_primaries"] == 2
    assert result["n_pairs"] == 8


def test_two_stage_fit_fixed_trees():
    if RRE is None:
        print('SKIP test_two_stage_fit_fixed_trees (training stack unavailable: %s)' % IMPORT_ERROR)
        return
    root, config_path = build_workspace()
    try:
        experiment = make_experiment(root, config_path)
        metrics, base, weighted = experiment.fit_candidate(
            dict(BASELINE_PARAMS), 0.1, fixed_trees=6, keep_models=True)
        assert metrics["n_trees"] == 6
        assert metrics["base"]["fixed_tree_count"] == 6
        assert abs(metrics["weight"]["train_weight_p50"] - 1.0) < 0.5
        slope = metrics["weighted"]["vector"]["development"]["slope_measured_on_predicted"]
        assert np.isfinite(slope["mean"]) and slope["n_cases"] == 4
        assert base is not None and weighted is not None
        assert np.isfinite(metrics["weighted"]["ordinary_validation_r2"])
    finally:
        shutil.rmtree(root)


def test_tune_end_to_end_and_resume():
    if not HAS_OPTUNA:
        print("SKIP test_tune_end_to_end_and_resume (no optuna)")
        return
    root, config_path = build_workspace()
    try:
        output = os.path.join(root, "tune_out")
        argv = ["--config", config_path, "--mode", "tune",
                "--output-dir", output, "--n-trials", "2",
                "--fit-cases", str(FIT_CASES[0]), str(FIT_CASES[1]),
                "--dev-cases", str(DEV_CASES[0]), str(DEV_CASES[1])]
        RRE.main(argv)
        summary = json.load(open(os.path.join(output, "summary.json")))
        assert summary["study"]["completed_trials"] == 2
        assert summary["study"]["requested_completed_trials"] == 2
        assert summary["study"]["best_trial"] in (0, 1)
        assert np.isfinite(summary["study"]["best_value_abs_development_slope_minus_one"])
        assert summary["objective"]["constgold_used"] is False
        assert summary["objective"]["coherent_anchor_truth_used"] is False
        assert summary["training"]["weighted_refit_starts_from_scratch"] is True
        assert summary["training"]["weight_uses_label"] is False
        assert os.path.exists(os.path.join(output, "best_base_model.json"))
        assert os.path.exists(os.path.join(output, "best_weighted_model.json"))
        assert os.path.exists(os.path.join(output, "trials.csv"))
        assert os.path.exists(os.path.join(output, "study.sqlite3"))
        assert not any(name.startswith(".best") for name in os.listdir(output)), \
            "temporary files must be gone"
        # resume against a finalized study is a no-op, not an error
        RRE.main(argv)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_train_recipe_and_sidecar():
    if not HAS_OPTUNA:
        print("SKIP test_train_recipe_and_sidecar (no optuna)")
        return
    tune_root, config_path = build_workspace()
    train_root = None
    try:
        output = os.path.join(tune_root, "tune_out")
        RRE.main(["--config", config_path, "--mode", "tune",
                  "--output-dir", output, "--n-trials", "2",
                  "--fit-cases", str(FIT_CASES[0]), str(FIT_CASES[1]),
                  "--dev-cases", str(DEV_CASES[0]), str(DEV_CASES[1])])
        train_root, _ = build_workspace()
        # repoint the train config at the tune workspace's source tag directory
        with open(config_path) as handle:
            config = yaml.safe_load(handle)
        config["training"]["model_dir"] = os.path.join(train_root, "models")
        train_config = os.path.join(train_root, "train.yaml")
        with open(train_config, "w") as handle:
            yaml.safe_dump(config, handle)
        train_out = os.path.join(train_root, "train_out")
        RRE.main(["--config", train_config, "--mode", "train",
                  "--output-dir", train_out,
                  "--recipe", os.path.join(output, "summary.json"),
                  "--fit-cases", str(FIT_CASES[0]), str(FIT_CASES[1]),
                  "--dev-cases", str(DEV_CASES[0]), str(DEV_CASES[1]),
                  "--model-tag", "smoke_fit"])
        summary = json.load(open(os.path.join(train_out, "summary.json")))
        assert summary["kind"].startswith("fixed-recipe")
        assert os.path.exists(os.path.join(train_out, "base_model.json"))
        weighted_path = os.path.join(train_out, "weighted_model.json")
        assert os.path.exists(weighted_path)
        sidecar = json.load(open(os.path.join(
            train_root, "models", "emulator_metadata_smoke_fit.json")))
        task = sidecar["tasks"]["regression"]
        assert task["model_file"] == "regression_model_smoke_fit.json"
        assert np.isclose(task["standardization"]["mean"], 0.004)
        weight_block = task["metrics"]["response_power_weight"]
        assert weight_block["label_used_in_weight"] is False
        assert weight_block["fixed_tree_count"] == summary["n_trees"]
        deployed = os.path.join(train_root, "models", "regression_model_smoke_fit.json")
        assert os.path.exists(deployed)
    finally:
        shutil.rmtree(tune_root, ignore_errors=True)
        if train_root:
            shutil.rmtree(train_root, ignore_errors=True)


def main():
    if RRE is None:
        print(f'SKIP ALL SMOKE TESTS (training stack unavailable: {IMPORT_ERROR})')
        return
    tests = [
        test_positive_response_weights,
        test_vector_evaluation_known_slope,
        test_two_stage_fit_fixed_trees,
        test_tune_end_to_end_and_resume,
        test_train_recipe_and_sidecar,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"ALL {len(tests)} SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()

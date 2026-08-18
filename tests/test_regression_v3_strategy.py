"""Smoke tests for the V3 regression strategy integrated into train_emulator.py.

Runnable two ways:
  pytest tests/test_regression_v3_strategy.py      # from the repo root
  python tests/test_regression_v3_strategy.py      # standalone (sims1 has no pytest)

Synthesizes a tiny ordered half-shear pair catalogue + config and exercises the
positive-response weights, the case-balanced vector score, the pair-population
loader (official split, fit-population standardization, window coverage), the
two-stage fit, and the tune/train CLI end to end. CPU only (XGB_DEVICE=cpu).
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
for _path in (REPO_ROOT,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("XGB_DEVICE", "cpu")

try:
    from scripts import train_emulator as TE
    IMPORT_ERROR = None
except Exception as error:  # sims1-only stack (train_emulator imports optuna)
    TE = None
    IMPORT_ERROR = error

if TE is not None:
    # Keep the smoke fits tiny regardless of entry point.
    TE.REGRESSION_MAX_ROUNDS = 80
    TE.REGRESSION_EARLY_STOPPING = 10

DEV_CASES = (0, 3)
FIT_CASES = (4, 5)


def build_workspace():
    root = tempfile.mkdtemp(prefix="v3_strategy_smoke_")
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
            "model_dir": model_dir, "model_tag": "smoke",
            "regression_cuts": [[13, 29], [18, 25.8], [0.0, 10.0], [0.5, 1.5], [0, 10]],
            "regression_fit_cases": list(FIT_CASES),
            "regression_dev_cases": list(DEV_CASES),
            "rescale": {"pixel_rms": 0.312, "pixel_size": 0.2, "zero_mag": 30,
                        "psf_fwhm": 0.73, "moffat_beta": 2.224},
        },
    }
    config_path = os.path.join(root, "smoke.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle)
    return root, config_path, frame


def load_data(config_path):
    from blendemu.config import load_config
    return TE.load_regression_data(load_config(config_path))


def _skip(name):
    if TE is None:
        print("SKIP %s (training stack unavailable: %s)" % (name, IMPORT_ERROR))
        return True
    return False


def test_positive_response_weights():
    if _skip("test_positive_response_weights"):
        return
    base_train = np.array([0.0, 1.0, 2.0, 10.0])
    base_validation = np.array([1.0, 3.0])
    alpha, cap = 0.1, 50.0
    train_w, val_w, meta = TE.positive_response_weights(base_train, base_validation,
                                                        alpha, cap)
    mean_power = float(np.mean(np.maximum(base_train, 0.0) ** 2))
    expected = 1.0 + alpha * np.minimum(np.maximum(base_train, 0.0) ** 2 / mean_power, cap)
    expected = expected / expected.mean()
    np.testing.assert_allclose(train_w, expected.astype(np.float32), rtol=1e-6)
    assert abs(float(train_w.mean()) - 1.0) < 1e-6
    assert meta["train_weight_max"] <= (1.0 + alpha * cap) / (1.0 + alpha) + 1e-9
    assert val_w.shape == (2,) and np.all(val_w > 0)


def test_vector_evaluation_known_slope():
    if _skip("test_vector_evaluation_known_slope"):
        return
    case = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    primary = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    angle = np.array([0.0, 0.0, 45.0, 45.0, 0.0, 0.0, 45.0, 45.0])
    k = np.array([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
    label = k / 2.0 * np.cos(np.deg2rad(2 * angle)) + k / 2.0 * np.sin(np.deg2rad(2 * angle))
    null = np.zeros(8)
    result = TE.VectorEvaluation(case, primary, angle, label, null).score(np.ones(8), (0, 1))
    np.testing.assert_allclose(
        [result["slope_mean"], result["orthogonal_null_mean"]],
        [0.75, 0.0], rtol=1e-10)
    assert result["n_primaries"] == 2 and result["n_pairs"] == 8


def test_load_regression_data_split_and_windows():
    if _skip("test_load_regression_data_split_and_windows"):
        return
    root, config_path, frame = build_workspace()
    try:
        data = load_data(config_path)
        fit_rows = int(((frame.case >= FIT_CASES[0]) & (frame.case <= FIT_CASES[1])).sum())
        dev_rows = int(((frame.case >= DEV_CASES[0]) & (frame.case <= DEV_CASES[1])).sum())
        assert data["n_train"] + data["n_validation"] == fit_rows
        assert data["n_dev"] == dev_rows
        assert data["n_train"] == int(0.8 * fit_rows) or abs(
            data["n_train"] - 0.8 * fit_rows) <= 1
        # standardization is computed on the fit-population labels
        fit_mask = (frame.case >= FIT_CASES[0]) & (frame.case <= FIT_CASES[1])
        fit_label = frame.loc[fit_mask, "delta_et1"].to_numpy() / 0.2
        assert np.isclose(data["y_mean"], fit_label.mean())
        assert np.isclose(data["y_std"], fit_label.std(ddof=1))
        # same seed -> identical split mask
        again = load_data(config_path)
        assert again["n_train"] == data["n_train"]
        np.testing.assert_array_equal(data["x_train"], again["x_train"])
        assert data["dm_train"].num_row() == data["n_train"]
        assert data["dm_dev"].num_row() == data["n_dev"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_regression_data_rejects_gap_cases():
    if _skip("test_load_regression_data_rejects_gap_cases"):
        return
    root, config_path, frame = build_workspace()
    try:
        with open(config_path) as handle:
            config = yaml.safe_load(handle)
        # introduce a gap: rows of case 3 (dev) fall outside both windows
        config["training"]["regression_dev_cases"] = [0, 2]
        gap_path = os.path.join(root, "gap.yaml")
        with open(gap_path, "w") as handle:
            yaml.safe_dump(config, handle)
        from blendemu.config import load_config
        try:
            TE.load_regression_data(load_config(gap_path))
            raise AssertionError("expected RuntimeError for case between windows")
        except RuntimeError as error:
            assert "3" in str(error)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fit_two_stage_fixed_trees():
    if _skip("test_fit_two_stage_fixed_trees"):
        return
    root, config_path, _ = build_workspace()
    try:
        data = load_data(config_path)
        params = {"subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.3,
                  "max_depth": 4, "min_child_weight": 1, "reg_lambda": 1.0,
                  "reg_alpha": 0.1, "gamma": 0.0}
        metrics, base, weighted = TE.fit_two_stage(data, params, 0.1,
                                                   fixed_trees=6, keep_models=True)
        assert metrics["n_trees"] == 6
        assert metrics["base"]["best_validation_rmse_standardized"] is None
        assert abs(metrics["weight"]["train_weight_p50"] - 1.0) < 0.5
        assert np.isfinite(metrics["weighted"]["dev_slope"]["slope_mean"])
        assert np.isfinite(metrics["weighted"]["ordinary_validation_r2"])
        assert base is not None and weighted is not None
        # weights are label-free: identical inputs give identical boosters
        metrics2, _, weighted2 = TE.fit_two_stage(data, params, 0.1,
                                                  fixed_trees=6, keep_models=True)
        assert metrics2["weight"] == metrics["weight"]
        assert np.array_equal(weighted.predict(data["dm_dev"]),
                              weighted2.predict(data["dm_dev"]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_cli(config_path, extra):
    argv = sys.argv
    try:
        sys.argv = ["train_emulator.py", "--config", config_path] + extra
        TE.main()
    finally:
        sys.argv = argv


def test_cli_tune_and_train_end_to_end():
    if _skip("test_cli_tune_and_train_end_to_end"):
        return
    root, config_path, _ = build_workspace()
    try:
        _run_cli(config_path, ["--mode", "tune", "--task", "regression",
                               "--n-trials", "2"])
        model_dir = os.path.join(root, "models")
        assert os.path.exists(os.path.join(model_dir, "regression_model_smoke.json"))
        sidecar = json.load(open(
            os.path.join(model_dir, "emulator_metadata_smoke.json")))
        task = sidecar["tasks"]["regression"]
        assert task["model_file"] == "regression_model_smoke.json"
        assert task["standardization"]["std"] > 0
        block = task["metrics"]["response_power_weight"]
        assert block["label_used_in_weight"] is False
        assert block["refit_uses_base_selected_tree_count"] is True
        assert task["metrics"]["best_trees"] >= 1
        # train-only rerun through the no-optuna default-params path
        _run_cli(config_path, ["--mode", "train", "--task", "regression",
                               "--no-optuna"])
        sidecar = json.load(open(
            os.path.join(model_dir, "emulator_metadata_smoke.json")))
        block = sidecar["tasks"]["regression"]["metrics"]["response_power_weight"]
        assert np.isclose(block["alpha"],
                          TE.DEFAULT_PARAMS["regression"]["response_weight_alpha"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    if TE is None:
        print("SKIP ALL SMOKE TESTS (training stack unavailable: %s)" % IMPORT_ERROR)
        return
    tests = [
        test_positive_response_weights,
        test_vector_evaluation_known_slope,
        test_load_regression_data_split_and_windows,
        test_load_regression_data_rejects_gap_cases,
        test_fit_two_stage_fixed_trees,
        test_cli_tune_and_train_end_to_end,
    ]
    for test in tests:
        test()
        print("PASS %s" % test.__name__, flush=True)
    print("ALL %d SMOKE TESTS PASSED" % len(tests), flush=True)


if __name__ == "__main__":
    main()

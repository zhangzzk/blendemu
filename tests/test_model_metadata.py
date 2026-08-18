import json

from blendemu.inference import BlendingPredictor
from blendemu import inference


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}


class FakeBooster:
    def __init__(self, params):
        self.params = params
        self.feature_names = ["feature"]
        self.loaded_path = None

    def load_model(self, path):
        self.loaded_path = path


def test_load_prefers_json_metadata_sidecar(tmp_path, monkeypatch):
    metadata = {
        "tasks": {
            "classification": {
                "model_file": "cla.json",
                "boundary": [[1.0, 2.0]],
            },
            "regression": {
                "model_file": "reg.json",
                "boundary": [[3.0, 4.0]],
                "standardization": {"mean": 0.5, "std": 2.0},
            },
            "self_response": {
                "model_file": "self.json",
                "boundary": [[5.0, 6.0]],
                "standardization": {"mean": 0.7, "std": 3.0},
            },
        },
    }
    (tmp_path / "emulator_metadata_test.json").write_text(json.dumps(metadata))
    for filename in ("cla.json", "reg.json", "self.json"):
        (tmp_path / filename).write_text("{}")

    monkeypatch.setattr(inference.xgb, "Booster", FakeBooster)

    predictor = BlendingPredictor.load(
        tmp_path,
        tag="test",
        conditions=CONDITIONS,
        device="cpu",
    )

    assert predictor.bst_cla.loaded_path == str(tmp_path / "cla.json")
    assert predictor.bst_reg.loaded_path == str(tmp_path / "reg.json")
    assert predictor.bst_self.loaded_path == str(tmp_path / "self.json")
    assert predictor.y_mean == 0.5
    assert predictor.y_std == 2.0
    assert predictor.y_mean_self == 0.7
    assert predictor.y_std_self == 3.0
    assert predictor.boundaries_cla.tolist() == [[1.0, 2.0]]
    assert predictor.boundaries_reg.tolist() == [[3.0, 4.0]]
    assert predictor.boundaries_self.tolist() == [[5.0, 6.0]]

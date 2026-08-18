import pytest

from blendemu import catalog
from blendemu.config import (
    load_config,
    resolve_simulation_set,
    simulation_case_window,
    simulation_shear_scale,
)
from scripts import run_pipeline


def _load_config_with_detection(tmp_path, detection_yaml):
    base_config = tmp_path / "base.ini"
    base_config.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
catalog:
  path: /tmp/catalog.feather
  mag_cut: 27
simulation:
  output_path: /tmp/blendemu-out
  base_config: base.ini
noise:
  csv_path: /tmp/noise.csv
catalogues:
  detection:
{detection_yaml}
"""
    )
    return load_config(str(config_path))


def test_legacy_detection_radius_degrees_convert_after_defaults(tmp_path):
    cfg = _load_config_with_detection(
        tmp_path,
        "    r_max_deg: 0.0008333333333333334\n"
        "    r_min_deg: 0.0001\n"
        "    k: 2\n",
    )

    detection = cfg["catalogues"]["detection"]
    assert detection["r_max"] == pytest.approx(3.0)
    assert detection["r_min"] == pytest.approx(0.36)
    assert "r_max_deg" not in detection
    assert "r_min_deg" not in detection


def test_detection_radius_arcsec_keys_win_over_legacy_degrees(tmp_path):
    cfg = _load_config_with_detection(
        tmp_path,
        "    r_max: 4\n"
        "    r_max_deg: 0.0008333333333333334\n"
        "    r_min: 1\n"
        "    r_min_deg: 0.0001\n"
        "    k: 2\n",
    )

    detection = cfg["catalogues"]["detection"]
    assert detection["r_max"] == 4
    assert detection["r_min"] == 1
    assert "r_max_deg" not in detection
    assert "r_min_deg" not in detection


def test_nested_simulation_sets_resolve_task_scales(tmp_path):
    base_config = tmp_path / "base.ini"
    base_config.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
catalog:
  path: /tmp/catalog.feather
simulation:
  output_path: /tmp/blendemu-out
  base_config: base.ini
  n_cases: 200
  case_offset: 5
  shear_values: [0.0, 0.1]
  response:
    n_cases: 100
    shear_values: [0.0, 0.2]
  self_response:
    n_cases: 80
    case_offset: 10
    shear_values: [0.0, 0.05]
noise:
  csv_path: /tmp/noise.csv
"""
    )

    cfg = load_config(str(config_path))
    response = resolve_simulation_set(cfg["simulation"], "response")
    self_response = resolve_simulation_set(cfg["simulation"], "self_response")

    assert simulation_case_window(response) == (5, 100)
    assert simulation_shear_scale(response) == pytest.approx(0.2)
    assert simulation_case_window(self_response) == (10, 80)
    assert simulation_shear_scale(self_response) == pytest.approx(0.05)


def test_pipeline_detection_defaults_to_self_response_set(tmp_path):
    base_config = tmp_path / "base.ini"
    base_config.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
catalog:
  path: /tmp/catalog.feather
simulation:
  output_path: /tmp/blendemu-out
  base_config: base.ini
  n_cases: 200
  shear_values: [0.0, 0.1]
  response:
    n_cases: 100
    shear_values: [0.0, 0.2]
  self_response:
    n_cases: 100
    shear_values: [0.0, 0.05]
noise:
  csv_path: /tmp/noise.csv
catalogues:
  detection:
    r_max: 3
    k: 2
"""
    )

    cfg = load_config(str(config_path))
    cat_cfg = cfg["catalogues"]
    self_sim_cfg = resolve_simulation_set(
        cfg["simulation"],
        "self_response",
        overrides=cat_cfg["self_response"],
    )

    assert run_pipeline._detection_case_window(cat_cfg["detection"], self_sim_cfg) == (0, 100)
    assert run_pipeline._detection_shear_label(cat_cfg["detection"], self_sim_cfg) == "0.05"


def test_constant_shear_mode_config_loads_defaults(tmp_path):
    base_config = tmp_path / "base.ini"
    base_config.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
catalog:
  path: /tmp/catalog.feather
simulation:
  output_path: /tmp/blendemu-out
  base_config: base.ini
  n_cases: 50
  shear_mode: constant
  constant_shear: 0.02
  constant_two_sided: true
noise:
  csv_path: /tmp/noise.csv
"""
    )

    cfg = load_config(str(config_path))
    sim = cfg["simulation"]
    assert sim["shear_mode"] == "constant"
    assert sim["constant_shear"] == 0.02
    assert sim["constant_two_sided"] is True
    # angle default fills in
    assert sim["constant_shear_angle"] == 0.0
    # constant_shear catalogue section defaults are present
    cs = cfg["catalogues"]["constant_shear"]
    assert cs["shape_suffix"] == "_all"
    assert cs["output_prefix"] == "constant_shear_catalogue"
    assert cs["response_output_prefix"] == "constant_response_catalogue"


def test_constant_shear_values_and_work_items():
    sim_cfg = {
        "case_offset": 0,
        "n_cases": 3,
        "constant_shear": 0.02,
        "constant_two_sided": True,
    }
    assert run_pipeline._constant_shear_values(sim_cfg) == [0.02, -0.02]

    items = run_pipeline._constant_work_items(sim_cfg)
    assert [it["shear_label"] for it in items] == ["0.02", "-0.02"]
    assert all(it["n_cases"] == 3 for it in items)

    sim_cfg_single = dict(sim_cfg, constant_two_sided=False)
    assert run_pipeline._constant_shear_values(sim_cfg_single) == [0.02]


def test_constant_step_map_drops_secondary_and_train_steps():
    step_map = run_pipeline.build_step_map("constant")
    assert set(step_map) == {"1", "2", "3", "4"}
    assert step_map["3"][0] == "measure_all"
    assert step_map["4"][0] == "constant_catalogs"
    assert run_pipeline.parse_steps("all", shear_mode="constant") == ["1", "2", "3", "4"]
    assert run_pipeline.parse_steps("all", shear_mode="half") == [
        "1", "2", "3", "3b", "4", "4b", "5",
    ]


def test_sim_config_writer_applies_yaml_driven_overrides(tmp_path):
    base_config = tmp_path / "base.ini"
    base_config.write_text(
        """
[Paths]
out_dir = /old/out
tmp_dir = /old/tmp

[GalInfo]
cata_file = /old/cat.feather
mag_cut = 0, 25  # template value

[NoiseInfo]
cata_file = /old/noise.csv

[ImSim]
survey = one_tile
pixel_scale_list = 0.1

[SExtractor]
pixel_scale = 0.1

[CrossMatch]
mag_faint_cut = 24
"""
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_config = catalog.write_config_file(
        suffix=3,
        g=0.2,
        path=str(output_dir),
        base_config_path=str(base_config),
        file_name_cat="/cats/gals3_0.2.feather",
        survey="simple_0.01sqdeg",
        mag_cut=29,
        crossmatch_mag_faint_cut=29,
        pixel_scale=0.2,
    )

    text = (tmp_path / "out" / "sim_config_case3_0.2.ini").read_text()
    assert output_config == str(tmp_path / "out" / "sim_config_case3_0.2.ini")
    assert f"out_dir = {tmp_path / 'out' / 'case3_0.2'}" in text
    assert f"tmp_dir = {tmp_path / 'out' / 'case3_0.2_tmp'}" in text
    assert "cata_file = /cats/gals3_0.2.feather" in text
    assert f"cata_file = {tmp_path / 'out' / 'noise.csv'}" in text
    assert "mag_cut = 0, 29  # template value" in text
    assert "mag_faint_cut = 29" in text
    assert "pixel_scale_list = 0.2" in text
    assert "pixel_scale = 0.2" in text
    assert "survey = simple_0.01sqdeg" in text

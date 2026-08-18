"""End-to-end unit tests for the constant-shear catalogue builders.

These fabricate the minimal MultiBand_ImSim directory layout (input, Shapes,
CrossMatch feathers) under a tmp_path so the builders can run without GalSim /
SExtractor / MPI.
"""

import os

import numpy as np
import pandas as pd

from blendemu.catalog import SHEAR_COMPONENT_CONVENTION
from blendemu.response import retrieve_constant_shear, retrieve_constant_response

TILE = "tile180.0_-0.5"

# Four galaxies in two well-separated pairs (~5" within a pair, ~15" between
# pairs), so each has a unique nearest neighbour inside r_max=10" and none has a
# neighbour inside the 3" bright-neighbour guard.
RA = np.array([180.0, 180.0 + 5 / 3600, 180.0 + 20 / 3600, 180.0 + 25 / 3600])
DEC = np.zeros(4)
EXPECTED_NEIGHBOUR = [1, 0, 3, 2]


def _write_case(data_path, case, shear_label, ngmix_g1, gamma1, gamma2,
                ngmix_g2=None):
    """Write input + Shapes + CrossMatch feathers for one case/shear."""
    ngmix_g2 = np.zeros(4) if ngmix_g2 is None else np.asarray(ngmix_g2)
    case_dir = os.path.join(data_path, f"case{case}_{shear_label}", "real0", "catalogues")
    os.makedirs(os.path.join(case_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "Shapes"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "CrossMatch"), exist_ok=True)

    icat = pd.DataFrame({
        "RA_input": RA,
        "DEC_input": DEC,
        "redshift_input": [0.4, 0.5, 0.6, 0.7],
        "Re_input": [0.4, 0.5, 0.6, 0.3],
        "axis_ratio_input": [0.7, 0.8, 0.6, 0.9],
        "position_angle_input": [10.0, 20.0, 30.0, 40.0],
        "sersic_n_input": [1.0, 2.0, 1.5, 1.0],
        "r_input": [22.0, 23.0, 24.0, 21.0],
        "gamma1_input": np.full(4, gamma1),
        "gamma2_input": np.full(4, gamma2),
        "shear_component_convention": np.full(4, SHEAR_COMPONENT_CONVENTION),
    })
    icat.to_feather(os.path.join(case_dir, "input", f"gals_info_{TILE}.feather"))

    scat = pd.DataFrame({
        "X_WORLD": RA,
        "Y_WORLD": DEC,
        "X_IMAGE": np.arange(4) * 10.0 + 1.0,
        "Y_IMAGE": np.arange(4) * 10.0 + 1.0,
        "FLUX_AUTO": np.full(4, 1000.0),
        "FLUXERR_AUTO": np.full(4, 10.0),
        "NGMIX_G1": np.asarray(ngmix_g1, dtype=float),
        "NGMIX_G2": ngmix_g2.astype(float),
    })
    scat.to_feather(os.path.join(case_dir, "Shapes",
                                 f"shape_catalogue_detect_position_all_{TILE}.feather"))

    mcat = pd.DataFrame({
        "id_input": [0, 1, 2, 3],
        "id_detec": [1, 2, 3, 4],          # 1-indexed
        "distance_pixel_CM": np.zeros(4),
        "dmag_CM": np.zeros(4),
    })
    mcat.to_feather(os.path.join(case_dir, "CrossMatch", f"{TILE}_rot0_matched.feather"))


def test_retrieve_constant_shear_one_row_per_detection(tmp_path):
    ng1 = np.array([0.10, 0.05, -0.04, 0.08])
    _write_case(str(tmp_path), case=0, shear_label="0.02", ngmix_g1=ng1,
                gamma1=0.02, gamma2=0.0)

    df = retrieve_constant_shear(
        case=0, shear="0.02", r_max=10, r_min=0, k=5,
        data_path=str(tmp_path), tile_name=TILE,
    )

    assert len(df) == 4
    np.testing.assert_array_equal(df["input_index"].to_numpy(), [0, 1, 2, 3])
    # measured shape taken straight from NGMIX_*
    np.testing.assert_allclose(df["measured_e1"].to_numpy(), ng1)
    np.testing.assert_allclose(df["measured_e2"].to_numpy(), 0.0)
    # nearest neighbour attached
    assert df["neighbored"].all()
    np.testing.assert_array_equal(df["input_index_sec"].to_numpy().astype(int),
                                  EXPECTED_NEIGHBOUR)
    np.testing.assert_allclose(df["distance"].to_numpy(), 5.0, atol=1e-6)
    # applied shear stored; angle 0 -> pure g1
    np.testing.assert_allclose(df["applied_g1"].to_numpy(), 0.02)
    np.testing.assert_allclose(df["applied_g2"].to_numpy(), 0.0)
    np.testing.assert_allclose(df["shear_angle"].to_numpy(), 0.0)
    # neighbour input properties pulled from the right rows
    np.testing.assert_allclose(df["r_input_s"].to_numpy(),
                               np.array([23.0, 22.0, 21.0, 24.0]))


def test_retrieve_constant_response_per_object_response(tmp_path):
    g = 0.02
    e1p = np.array([0.10, 0.05, -0.04, 0.08])
    e1m = np.array([0.06, 0.07, -0.08, 0.02])
    _write_case(str(tmp_path), case=0, shear_label="0.02", ngmix_g1=e1p,
                gamma1=g, gamma2=0.0)
    _write_case(str(tmp_path), case=0, shear_label="-0.02", ngmix_g1=e1m,
                gamma1=-g, gamma2=0.0)

    df = retrieve_constant_response(
        case=0, shear_cases=["0.02", "-0.02"], r_max=10, r_min=0, k=5,
        data_path=str(tmp_path), tile_name=TILE,
    )

    assert len(df) == 4
    np.testing.assert_array_equal(df["input_index"].to_numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(df["measured_e1_plus"].to_numpy(), e1p)
    np.testing.assert_allclose(df["measured_e1_minus"].to_numpy(), e1m)
    # angle 0 -> et == e1; per-object response = (e_+ - e_-) / (2 g)
    np.testing.assert_allclose(df["delta_et"].to_numpy(), e1p - e1m)
    np.testing.assert_allclose(df["sum_et"].to_numpy(), e1p + e1m)
    np.testing.assert_allclose(df["response"].to_numpy(), (e1p - e1m) / (2 * g))
    # neighbours attached correctly
    np.testing.assert_array_equal(df["input_index_sec"].to_numpy().astype(int),
                                  EXPECTED_NEIGHBOUR)


def test_retrieve_constant_shear_flags_failed_measurements(tmp_path):
    # NGMIX sentinel -1 marks a failed measurement -> NaN-ed out.
    ng1 = np.array([0.10, -1.0, -0.04, 0.08])
    ng2 = np.array([0.01, -1.0, 0.00, -0.02])
    _write_case(str(tmp_path), case=1, shear_label="0.02", ngmix_g1=ng1,
                gamma1=0.02, gamma2=0.0, ngmix_g2=ng2)

    df = retrieve_constant_shear(
        case=1, shear="0.02", r_max=10, r_min=0, k=5,
        data_path=str(tmp_path), tile_name=TILE,
    )
    df = df.sort_values("input_index").reset_index(drop=True)
    assert np.isnan(df.loc[1, "measured_e1"])
    assert np.isnan(df.loc[1, "measured_e2"])
    assert not np.isnan(df.loc[0, "measured_e1"])

import numpy as np
import pandas as pd
import pytest

from blendemu.catalog import SHEAR_COMPONENT_CONVENTION, generate_catalog_realization
from blendemu.response import (
    LEGACY_SHEAR_COMPONENT_CONVENTION,
    _read_generated_catalog_convention,
    _input_id_lookup,
    _lookup_input_rows,
    _primary_secondary_rows,
    e2ang,
    spin2rot,
)


def _base_catalog():
    return pd.DataFrame({
        "sdss_r": [24.0, 25.0],
        "shape/sersic_n": [1.0, 2.0],
        "Re_arcsec": [0.4, 0.5],
        "BA": [0.7, 0.8],
        "redshift": [0.5, 0.8],
        "angle": [0.1, 0.2],
    })


def test_generate_catalog_realization_uses_usual_shear_basis():
    base = _base_catalog()
    g = 0.1

    # default shear_type is 'half': first half unsheared, second half sheared.
    cat = generate_catalog_realization(base, num_sim=8, seed=123, g=g)
    unsheared = cat.iloc[:4]
    sheared = cat.iloc[4:]

    np.testing.assert_allclose(unsheared["g1"], 0.0)
    np.testing.assert_allclose(unsheared["g2"], 0.0)
    np.testing.assert_allclose(np.hypot(sheared["g1"], sheared["g2"]), g)
    assert set(cat["shear_component_convention"]) == {SHEAR_COMPONENT_CONVENTION}


def test_generate_catalog_realization_constant_shears_all_galaxies():
    base = _base_catalog()
    g = 0.02

    cat = generate_catalog_realization(
        base, num_sim=8, seed=123, g=g, shear_type='constant'
    )

    # Every galaxy gets the same fixed shear; angle 0 -> pure g1.
    np.testing.assert_allclose(cat["g1"], g)
    np.testing.assert_allclose(cat["g2"], 0.0)
    np.testing.assert_allclose(np.hypot(cat["g1"], cat["g2"]), g)


def test_generate_catalog_realization_constant_honours_shear_angle():
    base = _base_catalog()
    g = 0.02
    theta = np.deg2rad(30.0)

    cat = generate_catalog_realization(
        base, num_sim=6, seed=7, g=g, shear_type='constant', shear_angle=theta
    )

    np.testing.assert_allclose(cat["g1"], g * np.cos(2 * theta))
    np.testing.assert_allclose(cat["g2"], g * np.sin(2 * theta))


def test_generate_catalog_realization_antithetic_pair_shares_galaxies():
    base = _base_catalog()
    g = 0.02

    plus = generate_catalog_realization(
        base, num_sim=8, seed=123, g=g, shear_type='constant'
    )
    minus = generate_catalog_realization(
        base, num_sim=8, seed=123, g=-g, shear_type='constant'
    )

    # Same seed -> identical galaxies and positions, exactly negated shear.
    for col in ["cata_idx", "RA", "DEC", "redshift", "Re", "axis_ratio",
                "sersic_n", "r", "position_angle"]:
        np.testing.assert_array_equal(plus[col].to_numpy(), minus[col].to_numpy())
    np.testing.assert_allclose(plus["g1"], -minus["g1"])
    np.testing.assert_allclose(plus["g2"], -minus["g2"])


def test_generate_catalog_realization_rejects_unknown_shear_type():
    base = _base_catalog()
    with pytest.raises(ValueError):
        generate_catalog_realization(base, num_sim=4, seed=1, g=0.02,
                                     shear_type='bogus')


def test_e2ang_uses_usual_spin2_half_angle():
    theta = np.deg2rad(np.array([0.0, 30.0, 80.0]))
    e1 = np.cos(2.0 * theta)
    e2 = np.sin(2.0 * theta)

    mag, angle = e2ang(e1, e2)

    np.testing.assert_allclose(mag, 1.0)
    np.testing.assert_allclose(angle, theta)


def test_e2ang_supports_legacy_sin_cos_shear_basis():
    theta = np.deg2rad(np.array([0.0, 30.0, 80.0]))
    e1 = np.sin(2.0 * theta)
    e2 = np.cos(2.0 * theta)

    mag, angle = e2ang(e1, e2, convention=LEGACY_SHEAR_COMPONENT_CONVENTION)

    np.testing.assert_allclose(mag, 1.0)
    np.testing.assert_allclose(angle, theta)


def test_generated_catalogue_without_convention_uses_legacy_basis(tmp_path):
    pd.DataFrame({"index": [0, 1]}).to_feather(tmp_path / "gals3_0.1.feather")

    convention = _read_generated_catalog_convention(str(tmp_path), case=3, shear="0.1")

    assert convention == LEGACY_SHEAR_COMPONENT_CONVENTION


def test_spin2rot_projects_parallel_and_cross_to_angle():
    theta = np.deg2rad(np.array([0.0, 30.0, 80.0]))
    e1 = np.cos(2.0 * theta)
    e2 = np.sin(2.0 * theta)

    parallel, cross = spin2rot(e1, e2, theta)

    np.testing.assert_allclose(parallel, 1.0)
    np.testing.assert_allclose(cross, 0.0, atol=1.0e-15)


def test_input_id_lookup_handles_clipped_simulation_rows():
    icat = pd.DataFrame({
        "index_input": [2, 5, 8, 11],
        "RA_input": [0.0, 1.0, 2.0, 3.0],
    })

    lookup, input_ids = _input_id_lookup(icat)
    primary_rows, secondary_rows = _primary_secondary_rows(icat, total_input_count=12)

    np.testing.assert_array_equal(input_ids, [2, 5, 8, 11])
    np.testing.assert_array_equal(_lookup_input_rows(lookup, [8, 2]), [2, 0])
    np.testing.assert_array_equal(primary_rows, [0, 1])
    np.testing.assert_array_equal(secondary_rows, [2, 3])

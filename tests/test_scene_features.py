import numpy as np
import pandas as pd

from blendemu.scene_features import (
    SCENE_FLUX_INPUT_COLUMNS,
    add_scene_flux_features,
    pair_other_absolute_flux,
    pair_other_flux_ratios,
    scene_flux_ratios,
)


def test_scene_flux_ratios_assign_shells_and_exclude_self():
    # Relative to galaxy 0: equal-flux neighbours at 0.5, 2, and 5 arcsec.
    ra = np.array([0.0, 0.5 / 3600, 2.0 / 3600, 5.0 / 3600])
    dec = np.zeros(4)
    mag = np.full(4, 25.0)
    values = scene_flux_ratios(ra, dec, mag)
    np.testing.assert_allclose(values[:, 0], np.log10([2.0, 2.0, 2.0]))
    assert np.all(values >= 0)


def test_add_scene_flux_features_uses_input_only_schema():
    frame = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "r": [24.0]})
    output = add_scene_flux_features(frame)
    assert list(output.columns[-3:]) == list(SCENE_FLUX_INPUT_COLUMNS)
    np.testing.assert_array_equal(output.iloc[0, -3:].to_numpy(float), 0.0)


def test_pair_other_flux_ratios_excludes_the_named_secondary_only():
    # Two pair rows share one primary-level scene. Each row names a different
    # secondary, so a different shell contribution must be removed.
    mag_p = np.array([20.0, 20.0])
    mag_s = np.array([21.0, 22.0])
    ratio_s = 10.0 ** (-0.4 * (mag_s - mag_p))
    other = np.array([0.25, 0.50, 0.75])
    total_ratio = np.tile(other, (2, 1))
    total_ratio[0, 0] += ratio_s[0]
    total_ratio[1, 1] += ratio_s[1]

    result = pair_other_flux_ratios(
        np.log10(1.0 + total_ratio), mag_p, mag_s, np.array([0.5, 2.0]),
    )
    np.testing.assert_allclose(
        result, np.tile(np.log10(1.0 + other), (2, 1)), rtol=0, atol=1e-12,
    )


def test_pair_other_flux_ratios_rejects_a_missing_designated_secondary():
    total = np.zeros((1, 3), dtype=float)
    with np.testing.assert_raises_regex(ValueError, "lacks the designated secondary"):
        pair_other_flux_ratios(total, np.array([20.0]), np.array([20.0]), np.array([0.5]))


def test_pair_other_absolute_flux_removes_primary_normalization():
    mag_p = np.array([20.0, 22.0])
    mag_s = np.array([21.0, 23.0])
    secondary_ratio = 10.0 ** (-0.4 * (mag_s - mag_p))
    # Both rows contain exactly 100 counts of other light in every shell.
    primary_flux = 10.0 ** (-0.4 * (mag_p - 30.0))
    other_ratio = 100.0 / primary_flux
    total = np.repeat(other_ratio[:, None], 3, axis=1)
    total[np.arange(2), [0, 1]] += secondary_ratio
    result = pair_other_absolute_flux(
        np.log10(1.0 + total), mag_p, mag_s, np.array([0.5, 2.0]),
    )
    np.testing.assert_allclose(result, np.log10(101.0), rtol=0, atol=1e-10)


def test_pair_other_flux_can_leave_intrinsically_outside_secondary_untouched():
    total_ratio = np.array([[0.25, 0.5, 0.75]])
    result = pair_other_flux_ratios(
        np.log10(1.0 + total_ratio), np.array([20.0]), np.array([20.0]),
        np.array([10.2]), allow_outside=True,
    )
    np.testing.assert_allclose(result, np.log10(1.0 + total_ratio), rtol=0, atol=1e-12)

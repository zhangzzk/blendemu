import numpy as np
import pandas as pd

from blendemu import data_utils


def test_rescale_adds_requested_pair_ratio_coordinates():
    pairs = pd.DataFrame({
        "Re_input_p": [0.50, 0.80],
        "Re_input_s": [0.25, 0.20],
        "r_input_p": [23.0, 24.0],
        "r_input_s": [24.0, 22.0],
        "distance": [1.50, 2.00],
    })

    got = data_utils.rescale(pairs.copy())

    np.testing.assert_allclose(
        got["log10_re_sum_over_distance"],
        np.log10(np.array([(0.50 + 0.25) / 1.50, (0.80 + 0.20) / 2.00])),
    )
    np.testing.assert_allclose(
        got["log10_flux_s_over_flux_p"],
        np.array([-0.4, 0.8]),
    )
    # The older diagnostic feature has the opposite primary/secondary order.
    np.testing.assert_allclose(
        got["flux_ratio"],
        -got["log10_flux_s_over_flux_p"],
    )


def test_rescale_ratio_coordinates_use_nan_for_invalid_pairs():
    pairs = pd.DataFrame({
        "Re_input_p": [0.5, np.nan],
        "Re_input_s": [0.2, 0.3],
        "r_input_p": [23.0, 23.0],
        "r_input_s": [np.nan, 24.0],
        "distance": [0.0, 1.0],
    })

    got = data_utils.rescale(pairs.copy())

    assert got["log10_re_sum_over_distance"].isna().all()
    assert np.isnan(got.loc[0, "log10_flux_s_over_flux_p"])
    assert got.loc[1, "log10_flux_s_over_flux_p"] == -0.4

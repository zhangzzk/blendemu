import numpy as np
import pytest

from scripts.augment_response_scene_flux import composite_key, match_lookup


def test_composite_key_and_strict_match():
    keys = composite_key([1, 2], [7, 9])
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_equal(match_lookup(keys, values, [2, 1], [9, 7]), values[::-1])
    with pytest.raises(KeyError):
        match_lookup(keys, values, [2], [8])

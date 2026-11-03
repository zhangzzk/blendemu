import numpy as np
import xgboost as xgb

from scripts.prepare_disk_response_moment import response_gains
from scripts.train_disk_response_moment import moment_gradient


def test_gains_equal_the_physical_disk_map_derivative():
    shapes = np.array([[.2, .6], [.8, -.5], [0., 0.]])
    angles = np.array([12., 72., -24.])
    gain, cross = response_gains(shapes, angles)
    e = shapes[:, 0]+1j*shapes[:, 1]
    u = np.exp(2j*np.deg2rad(angles))
    h = 1e-6
    plus = (e+h*u)/(1+h*np.conj(u)*e)
    minus = (e-h*u)/(1-h*np.conj(u)*e)
    projected = (plus-minus)/(2*h)*np.conj(u)
    np.testing.assert_allclose(gain, projected.real, atol=1e-9, rtol=0)
    np.testing.assert_allclose(cross, projected.imag, atol=1e-9, rtol=0)
    assert (gain > 0).all()


def test_moment_gradient_fits_physical_mean_not_mean_inverse_label():
    gain, z = np.array([.2, 1.8]), np.array([.1, .3])
    mean, std = .12, .4
    b = z.sum()/gain.sum()
    class Matrix:
        def get_weight(self):
            return gain
        def get_label(self):
            return (z-gain*mean)/std
    gradient, hessian = moment_gradient(np.full(2, (b-mean)/std), Matrix())
    np.testing.assert_allclose(gradient.sum(), 0, atol=1e-14)
    np.testing.assert_array_equal(hessian, gain)
    assert not np.isclose(b, (z/gain).mean())


def test_xgboost_custom_objective_recovers_conditional_physical_moments():
    gain = np.array([.2, 1.8, .5, 1.5], dtype=np.float32)
    z = np.array([.1, .3, .9, .3], dtype=np.float32)
    mean, std = .1, .5
    matrix = xgb.DMatrix(np.array([[0.], [0.], [1.], [1.]]), label=(z-gain*mean)/std, weight=gain)
    model = xgb.train({'tree_method': 'hist', 'max_depth': 1, 'eta': 1., 'min_child_weight': 0,
                       'reg_lambda': 0., 'reg_alpha': 0., 'gamma': 0., 'base_score': 0.,
                       'nthread': 1, 'disable_default_eval_metric': True},
                      matrix, num_boost_round=1, obj=moment_gradient)
    predicted = model.predict(matrix)*std+mean
    np.testing.assert_allclose(predicted, [.2, .2, .6, .6], atol=1e-6, rtol=0)

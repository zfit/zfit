from __future__ import print_function, division, absolute_import

import pytest
import tensorflow as tf
import numpy as np

import zfit.core.basepdf
from zfit.core.pdf import Gauss
from zfit.core.parameter import FitParameter
import zfit.settings


mu_true = 1.4
sigma_true = 1.8
mu = FitParameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma = FitParameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
gauss_params1 = Gauss(mu=mu, sigma=sigma)


class TestGaussian(zfit.core.basepdf.BasePDF):
    def _func(self, value):
        return tf.exp((-(value - mu_true) ** 2) / (2 * sigma_true ** 2))  # non-normalized gaussian



def true_gaussian_func(x):
    return np.exp(- (x - mu_true) ** 2 / (2 * sigma_true ** 2))


test_gauss1 = TestGaussian()
test_gauss1.norm_range = -15., 18.
gauss_params1.norm_range = -15., 18.

init = tf.global_variables_initializer()


# class LimitTensor(tf.Tensor):
#     def __init__(self, *args, **kwargs):
#         super(LimitTensor, self).__init__(*args, **kwargs)
#         self.limits = None


def test_func():
    test_values = np.array([3., 11.3, -0.2, -7.82])
    with tf.Session() as sess:
        vals = test_gauss1.func(
            tf.convert_to_tensor(test_values, dtype=zfit.settings.fptype))
        vals = sess.run(vals)
        vals_gauss = gauss_params1.func(
            tf.convert_to_tensor(test_values, dtype=zfit.settings.fptype))
        # init = tf.global_variables_initializer()
        sess.run(init)
        vals_gauss = sess.run(vals_gauss)
    np.testing.assert_almost_equal(vals, true_gaussian_func(test_values))
    np.testing.assert_almost_equal(vals_gauss, true_gaussian_func(test_values))


def test_normalization():
    with tf.Session() as sess:
        sess.run(init)
        low, high = -15., 18.
        samples = tf.cast(np.random.uniform(low=low, high=high, size=1000000), dtype=tf.float64)
        samples.limits = low, high
        probs = test_gauss1.prob(samples)
        probs2 = gauss_params1.prob(samples)
        result = sess.run(probs)
        result2 = sess.run(probs2)
        result = np.average(result) * (high - low)
        result2 = np.average(result2) * (high - low)
        assert 0.95 < result < 1.05
        assert 0.95 < result2 < 1.05
        assert -0.10 < result - result2 < 0.10

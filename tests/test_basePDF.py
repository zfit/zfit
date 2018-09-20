from __future__ import print_function, division, absolute_import

import pytest
import tensorflow as tf
import numpy as np

import zfit.core.pdf


class TestGaussian(zfit.core.pdf.BasePDF):
    def _func(self, value):
        return tf.exp(-(value - 1.4) ** 2 / 1.8 ** 2)  # non-normalized gaussian


def true_gaussian_func(x):
    return np.exp(- (x - 1.4) ** 2 / 1.8 ** 2)


test_gauss1 = TestGaussian()
test_gauss1.norm_range = -15., 18.

init = tf.global_variables_initializer()


def test_func():
    test_values = np.array([3., 129., -0.2, -78.2])
    with tf.Session() as sess:
        vals = test_gauss1.func(test_values)
        vals = sess.run(vals)
    np.testing.assert_almost_equal(vals, true_gaussian_func(test_values))


def test_normalization():
    with tf.Session() as sess:
        sess.run(init)
        low, high = -15., 18.
        probs = test_gauss1.prob(
            tf.cast(np.random.uniform(low=low, high=high, size=100000), dtype=tf.float32))
        result = sess.run(probs)
        result = np.average(result) * (high - low)
        assert 0.95 < result < 1.05

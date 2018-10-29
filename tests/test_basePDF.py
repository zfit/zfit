from __future__ import print_function, division, absolute_import

import pytest
import tensorflow as tf
import numpy as np

import zfit.core.basepdf
from zfit.core.limits import Range
from zfit.pdfs.dist_tfp import Normal
from zfit.pdfs.basic import Gauss
from zfit.core.parameter import FitParameter
import zfit.settings

mu_true = 1.4
sigma_true = 1.8
low, high = -4.3, 1.9
mu = FitParameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma = FitParameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
gauss_params1 = Gauss(mu=mu, sigma=sigma, name="gauss_params1")


class TestGaussian(zfit.core.basepdf.BasePDF):

    def _unnormalized_prob(self, x):
        return tf.exp((-(x - mu_true) ** 2) / (2 * sigma_true ** 2))  # non-normalized gaussian


def true_gaussian_func(x):
    return np.exp(- (x - mu_true) ** 2 / (2 * sigma_true ** 2))


mu2 = FitParameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma2 = FitParameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
tf_gauss1 = tf.distributions.Normal(loc=mu2, scale=sigma2, name="tf_gauss1")
wrapped_gauss = zfit.core.basepdf.WrapDistribution(tf_gauss1)

test_gauss1 = TestGaussian(name="test_gauss1")
wrapped_normal1 = Normal(loc=mu2, scale=sigma2, name='wrapped_normal1')

init = tf.global_variables_initializer()

gaussian_dists = [test_gauss1, gauss_params1]


def test_func():
    test_values = np.array([3., 11.3, -0.2, -7.82])
    with tf.Session() as sess:
        test_values_tf = tf.convert_to_tensor(test_values, dtype=zfit.settings.types.float)

        for dist in gaussian_dists:
            vals = dist.unnormalized_prob(test_values_tf)
            sess.run(init)
            vals = sess.run(vals)
            np.testing.assert_almost_equal(vals, true_gaussian_func(test_values),
                                           err_msg="assert_almost_equal failed for ".format(
                                               dist.name))


def test_normalization():
    with tf.Session() as sess:
        sess.run(init)
        test_yield = 1524.3

        samples = tf.cast(np.random.uniform(low=low, high=high, size=100000), dtype=tf.float64)
        small_samples = tf.cast(np.random.uniform(low=low, high=high, size=10), dtype=tf.float64)
        for dist in gaussian_dists + [wrapped_gauss, wrapped_normal1]:
            with dist.temp_norm_range(Range.from_boundaries(low, high, dims=Range.FULL)):
                samples.limits = low, high
                print("Testing currently: ", dist.name)
                probs = dist.prob(samples)
                probs_small = dist.prob(small_samples)
                log_probs = dist.log_prob(small_samples)
                probs, log_probs = sess.run([probs, log_probs])
                probs = np.average(probs) * (high - low)
                assert probs == pytest.approx(1., rel=0.05)
                assert log_probs == pytest.approx(sess.run(tf.log(probs_small)), rel=0.001)
                dist.set_yield(tf.constant(test_yield, dtype=tf.float64))
                probs_extended = dist.prob(samples)
                result_extended = sess.run(probs_extended)
                result_extended = np.average(result_extended) * (high - low)
                assert result_extended == pytest.approx(test_yield, rel=0.05)


def test_sampling():
    with tf.Session() as sess:
        sess.run(init)
        sampled_from_gauss1 = sess.run(gauss_params1.sample(n_draws=1000, limits=(low, high)))
        assert max(sampled_from_gauss1) <= high
        assert min(sampled_from_gauss1) >= low

        sampled_gauss1_full = sess.run(gauss_params1.sample(n_draws=10000,
                                                            limits=(mu_true - abs(sigma_true) * 5,
                                                                    mu_true + abs(sigma_true) * 5)))
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert mu_sampled == pytest.approx(mu_true, rel=0.07)
        assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

def test_analytic_sampling():

    class SampleGauss(TestGaussian):
        pass

    SampleGauss.register_analytic_integral(func=lambda limits, params: 2 * limits.get_boundaries()[1][0][0],
                                           limits=(-float("inf"), None), dims=(0,))  # DUMMY!
    SampleGauss.register_inverse_analytic_integral(func=lambda x, params: x+1000.)

    gauss1 = SampleGauss()
    sample = gauss1.sample(n_draws=10000, limits=(2., 5.))
    with tf.Session() as sess:
        sample = sess.run(sample)

        assert 1004. <= min(sample)
        assert 10010. >= max(sample)


def test_multiple_limits():
    with tf.Session() as sess:
        sess.run(init)
        dims = (0,)
        simple_limits = (-3.2, 9.1)
        multiple_limits_lower = ((-3.2,), (1.1,), (2.1,))
        multiple_limits_upper = ((1.1,), (2.1,), (9.1,))
        multiple_limits_range = Range.from_boundaries(lower=multiple_limits_lower, upper=multiple_limits_upper,
                                                      dims=dims)
        integral_simp = gauss_params1.integrate(limits=simple_limits)
        integral_mult = gauss_params1.integrate(limits=multiple_limits_range)

        integral_simp, integral_mult = sess.run([integral_simp, integral_mult])
        assert integral_simp == pytest.approx(integral_mult, rel=1e-3)  # big tolerance as mc is used





def test_copy():
    new_gauss = gauss_params1.copy()
    assert new_gauss == gauss_params1

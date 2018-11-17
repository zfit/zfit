import pytest
import tensorflow as tf
import numpy as np

import zfit.core.basepdf
from zfit.core.limits import Range
import zfit.pdfs.dist_tfp
from zfit.pdfs.dist_tfp import Normal
from zfit.pdfs.basic import Gauss
from zfit.core.parameter import Parameter
import zfit.settings
from zfit import ztf

# from zfit.ztf import
test_values = np.array([3., 11.3, -0.2, -7.82])

mu_true = 1.4
sigma_true = 1.8
low, high = -4.3, 1.9
mu = Parameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma = Parameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
gauss_params1 = Gauss(mu=mu, sigma=sigma, name="gauss_params1")


class TestGaussian(zfit.core.basepdf.BasePDF):

    def _unnormalized_pdf(self, x):
        return tf.exp((-(x - mu_true) ** 2) / (2 * sigma_true ** 2))  # non-normalized gaussian


def true_gaussian_unnorm_func(x):
    return np.exp(- (x - mu_true) ** 2 / (2 * sigma_true ** 2))


def true_gaussian_grad(x):
    grad_mu = -0.199471140200716 * (2 * mu_true - 2 * x) * np.exp(
        -(-mu_true + x) ** 2 / (2 * sigma_true ** 2)) / sigma_true ** 3
    grad_sigma = -0.398942280401433 * np.exp(
        -(-mu_true + x) ** 2 / (2 * sigma_true ** 2)) / sigma_true ** 2 + 0.398942280401433 * (
                     -mu_true + x) ** 2 * np.exp(-(-mu_true + x) ** 2 / (2 * sigma_true ** 2)) / sigma_true ** 4
    return np.array((grad_mu, grad_sigma)).transpose()


mu2 = Parameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma2 = Parameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
mu3 = Parameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma3 = Parameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)
tf_gauss1 = tf.distributions.Normal(loc=mu2, scale=sigma2, name="tf_gauss1")
wrapped_gauss = zfit.pdfs.dist_tfp.WrapDistribution(tf_gauss1)

gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3)

test_gauss1 = TestGaussian(name="test_gauss1")
wrapped_normal1 = Normal(mu=mu2, sigma=sigma2, name='wrapped_normal1')

init = tf.global_variables_initializer()

gaussian_dists = [test_gauss1, gauss_params1]


def test_gradient():
    random_vals = np.random.normal(2., 4., size=5)
    random_vals = np.array([1, 4])
    with tf.Session() as sess:
        sess.run(init)
        tensor_grad = gauss3.gradient(x=random_vals, norm_range=(-np.infty, np.infty))
        random_vals_eval = sess.run(tensor_grad)
        assert random_vals_eval == pytest.approx(true_gaussian_grad(random_vals), rel=1e-5)


def test_func():
    test_values = np.array([3., 11.3, -0.2, -7.82])
    with tf.Session() as sess:
        test_values_tf = ztf.convert_to_tensor(test_values, dtype=zfit.settings.types.float)

        for dist in gaussian_dists:
            vals = dist.unnormalized_pdf(test_values_tf)
            sess.run(init)
            vals = sess.run(vals)
            np.testing.assert_almost_equal(vals, true_gaussian_unnorm_func(test_values),
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
                probs = dist.pdf(samples)
                probs_small = dist.pdf(small_samples)
                log_probs = dist.log_pdf(small_samples)
                probs, log_probs = sess.run([probs, log_probs])
                probs = np.average(probs) * (high - low)
                assert probs == pytest.approx(1., rel=0.05)
                assert log_probs == pytest.approx(sess.run(tf.log(probs_small)), rel=0.001)
                dist.set_yield(tf.constant(test_yield, dtype=tf.float64))
                probs_extended = dist.pdf(samples)
                result_extended = sess.run(probs_extended)
                result_extended = np.average(result_extended) * (high - low)
                assert result_extended == pytest.approx(test_yield, rel=0.05)


def test_sampling():
    with tf.Session() as sess:
        sess.run(init)
        n_draws = 1000
        sample_tensor = gauss_params1.sample(n_draws=n_draws, limits=(low, high))
        sampled_from_gauss1 = sess.run(sample_tensor)
        print(sampled_from_gauss1)
        assert max(sampled_from_gauss1[0]) <= high
        assert min(sampled_from_gauss1[0]) >= low
        assert n_draws == len(sampled_from_gauss1[0])

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
    SampleGauss.register_inverse_analytic_integral(func=lambda x, params: x + 1000.)

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

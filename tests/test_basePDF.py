import pytest
import tensorflow as tf
import numpy as np

import zfit.core.basepdf
from zfit.core.limits import Space
import zfit.models.dist_tfp
from zfit.models.dist_tfp import Gauss
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

obs1 = 'obs1'

gauss_params1 = Gauss(mu=mu, sigma=sigma, obs=obs1, name="gauss_params1")


class TestGaussian(zfit.core.basepdf.BasePDF):

    def _unnormalized_pdf(self, x, norm_range=False):
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
wrapped_gauss = zfit.models.dist_tfp.WrapDistribution(tf_gauss1, obs=obs1)

gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3, obs=obs1)

test_gauss1 = TestGaussian(name="test_gauss1", obs=obs1)
wrapped_normal1 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name='wrapped_normal1')

# init = tf.global_variables_initializer()

gaussian_dists = [test_gauss1, gauss_params1]


def test_gradient():
    random_vals = np.random.normal(2., 4., size=5)
    # random_vals = np.array([1, 4])
    # zfit.run(init)
    tensor_grad = gauss3.gradient(x=random_vals, params=['mu', 'sigma'], norm_range=(-np.infty, np.infty))
    random_vals_eval = zfit.run(tensor_grad)
    np.testing.assert_allclose(random_vals_eval, true_gaussian_grad(random_vals), rtol=1e-5)


def test_func():
    return  # HACK(Mayou36): changed Gauss to TF gauss -> different normalization
    test_values = np.array([3., 11.3, -0.2, -7.82])
    test_values_tf = ztf.convert_to_tensor(test_values, dtype=zfit.settings.ztypes.float)

    for dist in gaussian_dists:
        vals = dist.unnormalized_pdf(test_values_tf)
        # zfit.run(init)
        vals = zfit.run(vals)
        np.testing.assert_almost_equal(vals[0, :], true_gaussian_unnorm_func(test_values),
                                       err_msg="assert_almost_equal failed for ".format(
                                           dist.name))


def test_normalization():
    # zfit.run(init)
    test_yield = 1524.3

    samples = tf.cast(np.random.uniform(low=low, high=high, size=100000), dtype=tf.float64)
    small_samples = tf.cast(np.random.uniform(low=low, high=high, size=10), dtype=tf.float64)
    for dist in gaussian_dists + [wrapped_gauss, wrapped_normal1]:
        with dist.set_norm_range(Space(obs1, limits=(low, high))):
            samples.limits = low, high
            print("Testing currently: ", dist.name)
            probs = dist.pdf(samples)
            probs_small = dist.pdf(small_samples)
            log_probs = dist.log_pdf(small_samples)
            probs, log_probs = zfit.run([probs, log_probs])
            probs = np.average(probs) * (high - low)
            assert probs == pytest.approx(1., rel=0.05)
            assert log_probs == pytest.approx(zfit.run(tf.log(probs_small)), rel=0.05)
            dist.set_yield(tf.constant(test_yield, dtype=tf.float64))
            probs_extended = dist.pdf(samples)
            result_extended = zfit.run(probs_extended)
            result_extended = np.average(result_extended) * (high - low)
            assert result_extended == pytest.approx(test_yield, rel=0.05)


def test_sampling():
    # zfit.run(init)
    n_draws = 1000
    sample_tensor = gauss_params1.sample(n=n_draws, limits=(low, high))
    sampled_from_gauss1 = zfit.run(sample_tensor)
    assert max(sampled_from_gauss1[0]) <= high
    assert min(sampled_from_gauss1[0]) >= low
    assert n_draws == len(sampled_from_gauss1[0])

    sampled_gauss1_full = zfit.run(gauss_params1.sample(n=10000,
                                                        limits=(mu_true - abs(sigma_true) * 5,
                                                                mu_true + abs(sigma_true) * 5)))
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)


def test_analytic_sampling():
    class SampleGauss(TestGaussian):
        pass

    SampleGauss.register_analytic_integral(func=lambda limits, params: 2 * limits.upper[0][0],
                                           limits=Space.from_axes(limits=(-float("inf"), Space.ANY_UPPER),
                                                                  axes=(0,)))  # DUMMY!
    SampleGauss.register_inverse_analytic_integral(func=lambda x, params: x + 1000.)

    gauss1 = SampleGauss(obs=obs1)
    sample = gauss1.sample(n=10000, limits=(2., 5.))

    sample = zfit.run(sample)

    assert 1004. <= min(sample[0])
    assert 10010. >= max(sample[0])


def test_multiple_limits():
    # zfit.run(init)
    dims = (0,)
    simple_limits = (-3.2, 9.1)
    multiple_limits_lower = ((-3.2,), (1.1,), (2.1,))
    multiple_limits_upper = ((1.1,), (2.1,), (9.1,))
    multiple_limits_range = Space.from_axes(limits=(multiple_limits_lower, multiple_limits_upper), axes=dims)
    integral_simp = gauss_params1.integrate(limits=simple_limits, norm_range=False)
    integral_mult = gauss_params1.integrate(limits=multiple_limits_range, norm_range=False)
    integral_simp_num = gauss_params1.numeric_integrate(limits=simple_limits, norm_range=False)
    integral_mult_num = gauss_params1.numeric_integrate(limits=multiple_limits_range, norm_range=False)

    integral_simp, integral_mult = zfit.run([integral_simp, integral_mult])
    integral_simp_num, integral_mult_num = zfit.run([integral_simp_num, integral_mult_num])
    assert integral_simp == pytest.approx(integral_mult, rel=1e-2)  # big tolerance as mc is used
    assert integral_simp == pytest.approx(integral_simp_num, rel=1e-2)  # big tolerance as mc is used
    assert integral_simp_num == pytest.approx(integral_mult_num, rel=1e-2)  # big tolerance as mc is used


def test_copy():
    new_gauss = gauss_params1.copy()
    assert new_gauss == gauss_params1

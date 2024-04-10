#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import zfit
from zfit import z
import zfit.z.numpy as znp
from scipy.stats import gamma as gamma_scipy
import tensorflow as tf


gamma_true = 5.0
beta_true = 0.8
mu_true = 1.0

def create_gamma(gamma, beta, mu, limits):
    obs = zfit.Space("obs1", limits)
    gamma = zfit.pdf.Gamma(gamma=gamma_true, beta=beta, mu=mu, obs=obs)
    return gamma, obs


def test_gamma_pdf():
    gamma, _ = create_gamma(gamma=gamma_true, beta=beta_true, mu=mu_true, limits=(1, 10))
    assert pytest.approx(
        gamma_scipy.pdf(4.5, a=gamma_true, scale=beta_true, loc=mu_true), rel=1e-5
    ) == gamma.pdf(4.5, norm=False)
    test_values = tf.range(1, 10, 10_000)
    np.testing.assert_allclose(
        gamma.pdf(test_values, norm=False),
        gamma_scipy.pdf(test_values, a=gamma_true, scale=beta_true, loc=mu_true),
        rtol=1e-5,
    )
    assert gamma.pdf(test_values, norm=False) <= gamma.pdf(4.5, norm=False)

    sample = gamma.sample(10_00)
    assert all(np.isfinite(sample.value())), "Some samples from the gamma PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(1 <= sample.value(), sample.value() <= 10))


def test_gamma_integral():
    qgauss, obs = create_gamma(gamma=gamma_true, beta=beta_true, mu=mu_true, limits=(1, 10))
    full_interval_analytic = qgauss.analytic_integrate(obs, norm=False).numpy()
    full_interval_numeric = qgauss.numeric_integrate(obs, norm=False).numpy()
    scipy_full_inttegral = gamma_scipy.cdf(10, a=gamma_true, scale=beta_true, loc=mu_true) - gamma_scipy.cdf(
        1, a=gamma_true, scale=beta_true, loc=mu_true
    )
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_analytic
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_numeric

    analytic_integral = qgauss.analytic_integrate(limits=(3, 6), norm=False).numpy()
    numeric_integral = qgauss.numeric_integrate(limits=(3, 6), norm=False).numpy()
    scipy_integral = gamma_scipy.cdf(6, a=gamma_true, scale=beta_true, loc=mu_true) - gamma_scipy.cdf(
        3, a=gamma_true, scale=beta_true, loc=mu_true
    )
    assert pytest.approx(numeric_integral, 1e-5) == analytic_integral
    assert pytest.approx(scipy_integral, 1e-5) == analytic_integral

#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import zfit
from zfit import z
import zfit.z.numpy as znp
from scipy.stats import binom as binom_scipy
import tensorflow as tf


p_true = 0.3
mu_true = 1.0
n_true = 13

def create_binom(p, mu, limits):
    obs = zfit.Space("obs1", limits)
    binom = zfit.pdf.Binomial(n=n_true, p=p, mu=mu, obs=obs)
    return binom, obs


def test_binom_pdf():
    binom, _ = create_binom(p=p_true, mu=mu_true, limits=(2, 10))
    assert pytest.approx(
        binom_scipy.pdf(4, n=n_true, p=p_true, loc=mu_true), rel=1e-5
    ) == binom.pdf(4, norm=False)
    test_values = znp.linspace(1, 10, 10_000)
    test_values_ones = np.ones_like(test_values)
    np.testing.assert_allclose(
        binom.pdf(test_values, norm=False),
        binom_scipy.pdf(test_values,n=n_true, p=p_true, loc=mu_true),
        rtol=1e-5,
    )
    np.testing.assert_array_less(binom.pdf(test_values, norm=False), test_values_ones * binom.pdf(4.2, norm=False))

    sample = binom.sample(10_00)
    assert np.all(np.isfinite(sample.value())), "Some samples from the binom PDF are NaN or infinite"
    assert sample.n_events == 1000
    ones_like = np.ones_like(sample.value())
    np.testing.assert_array_less(1 * ones_like, sample.value())
    np.testing.assert_array_less(sample.value(), ones_like * 10)


def test_gamma_integral():
    qgauss, obs = create_binom(p=p_true, mu=mu_true, limits=(1, 10))
    full_interval_analytic = qgauss.analytic_integrate(obs, norm=False)
    full_interval_numeric = qgauss.numeric_integrate(obs, norm=False)
    scipy_full_inttegral = gamma_scipy.cdf(10, p=p_true, loc=mu_true) - gamma_scipy.cdf(
        1, p=p_true, loc=mu_true
    )
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_analytic
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_numeric

    analytic_integral = qgauss.analytic_integrate(limits=(3, 6), norm=False)
    numeric_integral = qgauss.numeric_integrate(limits=(3, 6), norm=False)
    scipy_integral = gamma_scipy.cdf(6, a=p_true, scale=beta_true, loc=mu_true) - gamma_scipy.cdf(
        3, a=p_true, scale=beta_true, loc=mu_true
    )
    assert pytest.approx(numeric_integral, 1e-5) == analytic_integral
    assert pytest.approx(scipy_integral, 1e-5) == analytic_integral

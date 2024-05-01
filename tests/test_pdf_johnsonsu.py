#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import zfit
from zfit import z
import zfit.z.numpy as znp
from scipy.stats import johnsonsu as johnsonsu_scipy
import tensorflow as tf

mu_true = 8.0
lambd_true = 3.0
gamma_true = 2.0
delta_true = 3.0

def create_johnsonsu(mu, lambd, gamma, delta, limits):
    obs = zfit.Space("obs1", limits)
    johnsonsu = zfit.pdf.JohnsonSU(mu=mu, lambd=lambd, gamma=gamma, delta=delta, obs=obs)
    return johnsonsu, obs


def test_johnsonsu_pdf():
    johnsonsu, _ = create_johnsonsu(mu=mu_true, lambd=lambd_true, gamma=gamma_true, delta=delta_true, limits=(1, 10))
    assert pytest.approx(
        johnsonsu_scipy.pdf(6, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true), rel=1e-5
    ) == johnsonsu.pdf(6, norm=False)
    test_values = znp.linspace(1, 10, 10_000)
    np.testing.assert_allclose(
        johnsonsu.pdf(test_values, norm=False),
        johnsonsu_scipy.pdf(test_values, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true),
        rtol=1e-5,
    )

    sample = johnsonsu.sample(10_00)
    assert np.all(np.isfinite(sample.value())), "Some samples from the JohnsonSU PDF are NaN or infinite"
    assert sample.n_events == 1000
    ones_like = np.ones_like(sample.value())
    np.testing.assert_array_less(1 * ones_like, sample.value())
    np.testing.assert_array_less(sample.value(), ones_like * 10)
    assert np.all(tf.logical_and(sample.value() >= 1, sample.value() <= 10))


def test_johnsonsu_integral():
    johnsonsu, obs = create_johnsonsu(mu=mu_true, lambd=lambd_true, gamma=gamma_true, delta=delta_true, limits=(1, 10))
    full_interval_analytic = johnsonsu.analytic_integrate(obs, norm=False)
    full_interval_numeric = johnsonsu.numeric_integrate(obs, norm=False)
    scipy_full_inttegral = johnsonsu_scipy.cdf(10, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true) - johnsonsu_scipy.cdf(
        1, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true
    )
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_analytic
    assert pytest.approx(scipy_full_inttegral, 1e-6) == full_interval_numeric

    analytic_integral = johnsonsu.analytic_integrate(limits=(5, 7), norm=False)
    numeric_integral = johnsonsu.numeric_integrate(limits=(5, 7), norm=False)
    scipy_integral = johnsonsu_scipy.cdf(7, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true) - johnsonsu_scipy.cdf(
        5, a=gamma_true, b=delta_true, loc=mu_true, scale=lambd_true
    )
    assert pytest.approx(numeric_integral, 1e-5) == analytic_integral
    assert pytest.approx(scipy_integral, 1e-5) == analytic_integral

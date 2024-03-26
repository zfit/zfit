#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import zfit
from zfit import z
import zfit.z.numpy as znp
from numba_stats import qgaussian as qgaussian_numba
import tensorflow as tf

mu_true = 0.5
sigma_true = 1.1
true_integral_dict = {1.00001: 0.9999782, 1.5: 0.9695, 2.0: 0.8063152, 2.5: 0.4805167, 2.9: 0.1078687, 2.99: 0.011044}


def create_qgaussian(q, mu, sigma, limits):
    obs = zfit.Space("obs1", limits)
    qgaussian = zfit.pdf.QGaussian(q=q, mu=mu, sigma=sigma, obs=obs)
    return qgaussian, obs


@pytest.mark.parametrize("q", [1.00001, 1.5, 2.0, 2.5, 2.9, 2.99])
def test_qgaussian_pdf(q):
    qgaussian, _ = create_qgaussian(q=q, mu=mu_true, sigma=sigma_true, limits=(-5, 5))
    assert qgaussian.pdf(0.5, norm=False).numpy().item() == pytest.approx(
        qgaussian_numba.pdf(0.5, q=q, mu=mu_true, sigma=sigma_true), rel=1e-5
    )
    np.testing.assert_allclose(
        qgaussian.pdf(tf.range(-5, 5, 10_000), norm=False).numpy(),
        qgaussian_numba.pdf(tf.range(-5, 5, 10_000), q=q, mu=mu_true, sigma=sigma_true),
        rtol=1e-5,
    )
    assert qgaussian.pdf(tf.range(-5, 5, 10_000), norm=False) <= qgaussian.pdf(0.5, norm=False)

    sample = qgaussian.sample(1000)
    tf.debugging.assert_all_finite(sample.value(), "Some samples from the qgaussian PDF are NaN or infinite")
    assert sample.n_events == 1000
    assert all(tf.logical_and(-5 <= sample.value(), sample.value() <= 5))


@pytest.mark.parametrize("q", [1.00001, 1.5, 2.0, 2.5, 2.9, 2.99])
def test_qgaussian_integral(q):
    qgaussian, obs = create_qgaussian(q=q, mu=mu_true, sigma=sigma_true, limits=(-5, 5))
    full_interval_analytic = qgaussian.analytic_integrate(obs, norm=False).numpy().item()
    full_interval_numeric = qgaussian.numeric_integrate(obs, norm=False).numpy().item()
    true_integral = true_integral_dict[q]
    numba_stats_full_integral = qgaussian_numba.cdf(5, q=q, mu=mu_true, sigma=sigma_true) - qgaussian_numba.cdf(
        -5, q=q, mu=mu_true, sigma=sigma_true
    )
    assert full_interval_analytic == pytest.approx(true_integral, 1e-4)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-4)
    assert full_interval_analytic == pytest.approx(numba_stats_full_integral, 1e-6)
    assert full_interval_numeric == pytest.approx(numba_stats_full_integral, 1e-6)

    analytic_integral = qgaussian.analytic_integrate(limits=(-1, 1), norm=False).numpy().item()
    numeric_integral = qgaussian.numeric_integrate(limits=(-1, 1), norm=False).numpy().item()
    numba_stats_integral = qgaussian_numba.cdf(1, q=q, mu=mu_true, sigma=sigma_true) - qgaussian_numba.cdf(
        -1, q=q, mu=mu_true, sigma=sigma_true
    )
    assert analytic_integral == pytest.approx(numeric_integral, 1e-5)
    assert analytic_integral == pytest.approx(numba_stats_integral, 1e-5)

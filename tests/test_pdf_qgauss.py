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
# obtained from the numba_stats package by numerically integrating the qgaussian PDF
true_integral_dict = {1.00001: 0.9999782, 1.5: 0.9695, 2.0: 0.8063152, 2.5: 0.4805167, 2.9: 0.1078687, 2.99: 0.011044}


def create_qgauss(q, mu, sigma, limits):
    obs = zfit.Space("obs1", limits)
    qgauss = zfit.pdf.QGauss(q=q, mu=mu, sigma=sigma, obs=obs)
    return qgauss, obs


@pytest.mark.parametrize("q", [1.00001, 1.5, 2.0, 2.5, 2.9, 2.99])
def test_qgauss_pdf(q):
    qgauss, _ = create_qgauss(q=q, mu=mu_true, sigma=sigma_true, limits=(-5, 5))
    assert qgauss.pdf(0.5, norm=False).numpy().item() == pytest.approx(
        qgaussian_numba.pdf(0.5, q=q, mu=mu_true, sigma=sigma_true), rel=1e-5
    )
    test_values = tf.range(-5, 5, 10_000)
    np.testing.assert_allclose(
        qgauss.pdf(test_values, norm=False).numpy(),
        qgaussian_numba.pdf(test_values, q=q, mu=mu_true, sigma=sigma_true),
        rtol=1e-5,
    )
    assert qgauss.pdf(test_values, norm=False) <= qgauss.pdf(0.5, norm=False)

    sample = qgauss.sample(1000)
    assert all(np.isfinite(sample.value())), "Some samples from the qgauss PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(-5 <= sample.value(), sample.value() <= 5))


@pytest.mark.parametrize("q", [1.00001, 1.5, 2.0, 2.5, 2.9, 2.99])
def test_qgauss_integral(q):
    qgauss, obs = create_qgauss(q=q, mu=mu_true, sigma=sigma_true, limits=(-5, 5))
    full_interval_analytic = qgauss.analytic_integrate(obs, norm=False).numpy()
    full_interval_numeric = qgauss.numeric_integrate(obs, norm=False).numpy()
    true_integral = true_integral_dict[q]
    numba_stats_full_integral = qgaussian_numba.cdf(5, q=q, mu=mu_true, sigma=sigma_true) - qgaussian_numba.cdf(
        -5, q=q, mu=mu_true, sigma=sigma_true
    )
    assert full_interval_analytic == pytest.approx(true_integral, 1e-4)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-4)
    assert full_interval_analytic == pytest.approx(numba_stats_full_integral, 1e-6)
    assert full_interval_numeric == pytest.approx(numba_stats_full_integral, 1e-6)

    analytic_integral = qgauss.analytic_integrate(limits=(-1, 1), norm=False).numpy()
    numeric_integral = qgauss.numeric_integrate(limits=(-1, 1), norm=False).numpy()
    numba_stats_integral = qgaussian_numba.cdf(1, q=q, mu=mu_true, sigma=sigma_true) - qgaussian_numba.cdf(
        -1, q=q, mu=mu_true, sigma=sigma_true
    )
    assert analytic_integral == pytest.approx(numeric_integral, 1e-5)
    assert analytic_integral == pytest.approx(numba_stats_integral, 1e-5)

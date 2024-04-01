#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import zfit
from zfit import z
import zfit.z.numpy as znp
import tensorflow as tf
from scipy import integrate

mu_true = 0.5
sigmal_true = 1.1
sigma_true = 1.5


def _numpy_bifurgauss_pdf(x, mu, sigmal, sigmar):
    norm = np.sqrt(2 / np.pi) / (sigmal + sigmar)
    if x < mu:
        return norm * np.exp(-0.5 * ((x - mu) / sigmal) ** 2)
    else:
        return norm * np.exp(-0.5 * ((x - mu) / sigmar) ** 2)

numpy_bifurgauss_pdf = np.vectorize(_numpy_bifurgauss_pdf, excluded=("mu", "sigmal", "sigmar"))


def create_bifurgauss(mu, sigmal, sigmar, limits):
    obs = zfit.Space("obs1", limits)
    bifurgauss = zfit.pdf.BifurGauss(mu=mu, sigmal=sigmal, sigmar=sigmar, obs=obs)
    return bifurgauss, obs



def test_bifurgauss_pdf():
    bifurgauss, obs = create_bifurgauss(mu=mu_true, sigmal=sigmal_true, sigmar=sigma_true, limits=(-5, 5))
    assert bifurgauss.pdf(0.5, norm=False).numpy().item() == pytest.approx(
        numpy_bifurgauss_pdf(0.5, mu=mu_true, sigmal=sigmal_true, sigmar=sigma_true), rel=1e-5
    )
    test_values = tf.range(-5, 5, 10_000)
    np.testing.assert_allclose(
        bifurgauss.pdf(test_values, norm=False).numpy(),
        numpy_bifurgauss_pdf(test_values, mu=mu_true, sigmal=sigmal_true, sigmar=sigma_true),
        rtol=1e-5,
    )
    assert bifurgauss.pdf(test_values, norm=False) <= bifurgauss.pdf(0.5, norm=False)

    sample = bifurgauss.sample(1000)
    assert all(np.isfinite(sample.value())), "Some samples from the bifurgauss PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(-5 <= sample.value(), sample.value() <= 5))



def test_bifurgauss_integral():
    bifurgauss, obs = create_bifurgauss(mu=mu_true, sigmal=sigmal_true, sigmar=sigma_true, limits=(-5, 5))
    full_interval_analytic = bifurgauss.analytic_integrate(obs, norm=False).numpy()
    full_interval_numeric = bifurgauss.numeric_integrate(obs, norm=False).numpy()
    true_integral = 0.998442
    numpy_full_integral = integrate.quad(
            numpy_bifurgauss_pdf, -5, 5, args=(mu_true, sigmal_true, sigma_true)
        )[0]
    assert full_interval_analytic == pytest.approx(true_integral, 1e-6)
    assert full_interval_numeric == pytest.approx(true_integral, 1e-6)
    assert full_interval_analytic == pytest.approx(numpy_full_integral, 1e-6)
    assert full_interval_numeric == pytest.approx(numpy_full_integral, 1e-6)

    analytic_integral = bifurgauss.analytic_integrate(limits=(-1, 1), norm=False).numpy()
    numeric_integral = bifurgauss.numeric_integrate(limits=(-1, 1), norm=False).numpy()
    numpy_integral = integrate.quad(
        numpy_bifurgauss_pdf, -1, 1, args=(mu_true, sigmal_true, sigma_true)
    )[0]
    assert analytic_integral == pytest.approx(numeric_integral, 1e-6)
    assert analytic_integral == pytest.approx(numpy_integral, 1e-6)


def test_equivalency_with_generalizedcb():
    bifurgauss, obs = create_bifurgauss(mu=mu_true, sigmal=sigmal_true, sigmar=sigma_true, limits=(-5, 5))
    generalized_cb = zfit.pdf.GeneralizedCB(mu=mu_true, sigmal=sigmal_true, alphal=100, nl=1, sigmar=sigma_true, alphar=100, nr=1, obs=obs)

    assert bifurgauss.pdf(0.5).numpy() == pytest.approx(generalized_cb.pdf(0.5).numpy(), rel=1e-5)
    test_values = tf.range(-5, 5, 10_000)
    np.testing.assert_allclose(
        bifurgauss.pdf(test_values).numpy(),
        generalized_cb.pdf(test_values).numpy(),
        rtol=1e-5,
    )

    assert bifurgauss.analytic_integrate(obs).numpy() == pytest.approx(generalized_cb.analytic_integrate(obs).numpy(), rel=1e-5)
    assert bifurgauss.numeric_integrate(obs).numpy() == pytest.approx(generalized_cb.numeric_integrate(obs).numpy(), rel=1e-5)
    assert bifurgauss.analytic_integrate(limits=(-1, 1)).numpy() == pytest.approx(generalized_cb.analytic_integrate(limits=(-1, 1)).numpy(), rel=1e-5)
    assert bifurgauss.numeric_integrate(limits=(-1, 1)).numpy() == pytest.approx(generalized_cb.numeric_integrate(limits=(-1, 1)).numpy(), rel=1e-5)

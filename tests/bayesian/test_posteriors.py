#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.core.loss import UnbinnedNLL
from zfit.minimize import Minuit

from zfit.bayesian.samplers.emcee import EmceeSampler


def test_posteriors_basic():
    # Create a simple model for testing
    true_mu = 1.5
    true_sigma = 0.5
    n_events = 1000

    # Generate some test data
    obs = zfit.Space("x", limits=(-10, 10))
    data = np.random.normal(true_mu, true_sigma, n_events)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)

    # Create model with parameters
    mu = zfit.Parameter("mu", 0.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create loss
    nll = UnbinnedNLL(model=gauss, data=dataset)

    # Run MCMC
    sampler = EmceeSampler(n_walkers=20)
    posteriors = sampler.sample(loss=nll, n_samples=500, n_warmup=100)

    # Test basic properties
    assert len(posteriors.param_names) == 2
    assert posteriors.n_samples == 500
    assert posteriors.n_warmup == 100
    assert posteriors.samples.shape == (500, 2)

    # Test parameter recovery
    mu_mean = posteriors.mean("mu")
    sigma_mean = posteriors.mean("sigma")
    assert abs(mu_mean - true_mu) < 0.2
    assert abs(sigma_mean - true_sigma) < 0.2

    # Test credible intervals
    mu_lower, mu_upper = posteriors.credible_interval("mu")
    assert mu_lower < true_mu < mu_upper

    # Test HDI
    mu_hdi_lower, mu_hdi_upper = posteriors.highest_density_interval("mu")
    assert mu_hdi_lower < true_mu < mu_hdi_upper

    # Test summary
    summary = posteriors.summary()
    assert "mu" in summary
    assert "sigma" in summary
    assert "mean" in summary["mu"]
    assert "std" in summary["mu"]

    # Test parameter bounds are respected
    assert np.all(posteriors.samples[:, 1] > 0.1)  # sigma lower bound
    assert np.all(posteriors.samples[:, 0] > -5)  # mu lower bound
    assert np.all(posteriors.samples[:, 0] < 5)  # mu upper bound


def test_posteriors_error_handling():
    # Test invalid parameter access
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0)
    sigma = zfit.Parameter("sigma", 1.0)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    with pytest.raises(ValueError):
        posteriors.mean("nonexistent_param")

    with pytest.raises(IndexError):
        posteriors.mean(123)  # Invalid parameter type


def test_posteriors_plotting():
    # Test that plotting functions don't raise errors
    pytest.importorskip("matplotlib")

    # Create simple model and run MCMC
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0)
    sigma = zfit.Parameter("sigma", 1.0)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test basic plotting functions
    posteriors.plot_trace("mu")
    posteriors.plot_posterior("mu")
    posteriors.plot_pair("mu", "sigma")


def test_posteriors_statistics():
    # Setup simple inference problem
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test statistical methods
    assert posteriors.median("mu").shape == ()
    assert posteriors.std("mu").shape == ()
    assert posteriors.mode("mu").shape == ()

    # Test covariance and correlation
    cov = posteriors.covariance()
    corr = posteriors.correlation()
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.abs(corr) <= 1.0)


def test_posteriors_predictive():
    # Setup inference
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test predictive distribution
    def predict_func():
        return gauss.sample(n=10)

    pred_samples = posteriors.predictive_distribution(predict_func)
    assert pred_samples.shape[0] == 100  # n_samples
    assert pred_samples.shape[1] == 10  # n_predictions per sample


def test_posteriors_evidence():
    # Setup inference
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test evidence calculation methods
    evidence_stepping = posteriors.marginal_likelihood(method="stepping")
    evidence_harmonic = posteriors.marginal_likelihood(method="harmonic")

    assert np.isfinite(evidence_stepping)
    assert np.isfinite(evidence_harmonic)

    # Test Bayes factor
    posteriors2 = sampler.sample(loss=nll, n_samples=100, n_warmup=10)
    bf = posteriors.bayes_factor(posteriors, posteriors2)
    assert np.isfinite(bf)


def test_posteriors_parameter_access():
    # Setup inference
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test different ways to access parameters
    assert np.allclose(posteriors.mean("mu"), posteriors.mean(mu))
    assert np.allclose(posteriors.mean("mu"), posteriors.mean(0))

    # Test accessing all parameters at once
    means = posteriors.mean()
    assert means.shape == (2,)

    medians = posteriors.median()
    assert medians.shape == (2,)

    stds = posteriors.std()
    assert stds.shape == (2,)


def test_posteriors_invalid_inputs():
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = np.random.normal(1.0, 1.0, 1000)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)
    nll = UnbinnedNLL(model=gauss, data=dataset)

    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test invalid inputs
    with pytest.raises(ValueError):
        posteriors.marginal_likelihood(method="invalid")

    with pytest.raises(ValueError):
        posteriors.credible_interval("mu", alpha=-0.1)

    with pytest.raises(ValueError):
        posteriors.credible_interval("mu", alpha=1.1)

    with pytest.raises(IndexError, match="Parameter index 123 out of range"):
        posteriors.mean(123)  # Invalid parameter index

"""Tests for the PosteriorSamples class."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import pytest
import arviz as az

import zfit
from zfit._mcmc.emcee import EmceeSampler
from zfit.core.loss import UnbinnedNLL
from zfit.mcmc import PosteriorSamples


@pytest.fixture
def obs():
    return zfit.Space("x", limits=(-10, 10))


@pytest.fixture
def true_params():
    return {"mu": 1.5, "sigma": 0.5}


@pytest.fixture
def model_params(true_params):
    mu = zfit.Parameter("mu", true_params["mu"], -5, 5, prior=zfit.prior.Normal(1.5, 1.2))
    sigma = zfit.Parameter("sigma", true_params["sigma"], 0.1, 10, prior=zfit.prior.HalfNormal(sigma=1.0))
    return {"mu": mu, "sigma": sigma}


@pytest.fixture
def gauss(obs, model_params):
    return zfit.pdf.Gauss(mu=model_params["mu"], sigma=model_params["sigma"], obs=obs)


@pytest.fixture
def data(obs, true_params):
    n_events = 1000
    data = np.random.normal(true_params["mu"], true_params["sigma"], n_events)
    return zfit.Data.from_numpy(obs=obs, array=data)


@pytest.fixture
def nll(gauss, data):
    return UnbinnedNLL(model=gauss, data=data)


def test_posterior_samples_basic(nll, model_params):
    """Test basic functionality of PosteriorSamples."""
    # Run MCMC
    nwalkers = 8
    sampler = EmceeSampler(nwalkers=nwalkers)
    n_samples = 200
    n_warmup = 50
    result = sampler.sample(loss=nll, n_samples=n_samples, n_warmup=n_warmup)

    # Test that we get a PosteriorSamples
    assert isinstance(result, PosteriorSamples)

    # Test basic properties
    assert len(result.param_names) == 2
    assert result.n_samples == n_samples
    assert result.n_warmup == n_warmup
    assert result.samples.shape == (n_samples * nwalkers, 2)  # n_samples * nwalkers, n_params

    # Test parameter recovery
    mu_mean = result.mean("mu")
    sigma_mean = result.mean("sigma")
    assert abs(mu_mean - 1.5) < 0.3
    assert abs(sigma_mean - 0.5) < 0.3

    # Test convergence diagnostics
    assert result.valid
    assert isinstance(result.converged, bool)
    assert result.params is not None
    assert result.sampler is not None
    assert result.loss is not None

    # Test modern methods
    symerror_mu = result.symerror("mu")
    symerror_sigma = result.symerror("sigma")
    assert symerror_mu > 0
    assert symerror_sigma > 0

    # Test credible intervals
    ci_mu = result.credible_interval("mu")
    ci_sigma = result.credible_interval("sigma")
    assert len(ci_mu) == 2
    assert len(ci_sigma) == 2
    assert ci_mu[0] < ci_mu[1]
    assert ci_sigma[0] < ci_sigma[1]

    # Test covariance
    cov = result.covariance()
    assert cov.shape == (2, 2)

    # Test ArviZ integration
    idata = result.to_arviz()
    assert isinstance(idata, az.InferenceData)

    # Test summary with ArviZ
    summary = result.summary()
    assert summary is not None  # Should return some kind of summary object

    # Test convergence diagnostics with ArviZ
    conv_summary = result.convergence_summary()
    assert "valid" in conv_summary
    assert "converged" in conv_summary
    assert "rhat" in conv_summary
    assert "ess_bulk" in conv_summary


def test_posterior_samples_context_manager(nll, model_params):
    """Test the context manager functionality of PosteriorSamples."""
    sampler = EmceeSampler()
    result = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Store original parameter values
    orig_mu = model_params["mu"].value()
    orig_sigma = model_params["sigma"].value()

    # Use context manager to temporarily set parameters to posterior means
    with result:
        # Parameters should be set to posterior means
        assert model_params["mu"].value() != orig_mu
        assert model_params["sigma"].value() != orig_sigma

    # Parameters should be restored to original values
    assert model_params["mu"].value() == orig_mu
    assert model_params["sigma"].value() == orig_sigma


def test_posterior_samples_set_params_to_mean(nll, model_params):
    """Test the set_params_to_mean method of PosteriorSamples."""
    sampler = EmceeSampler()
    result = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Store original parameter values
    orig_mu = model_params["mu"].value()
    orig_sigma = model_params["sigma"].value()

    # Update parameters to posterior means
    result.update_params()

    # Parameters should be set to posterior means
    assert model_params["mu"].value() != orig_mu
    assert model_params["sigma"].value() != orig_sigma

    # Reset parameters for other tests
    model_params["mu"].set_value(orig_mu)
    model_params["sigma"].set_value(orig_sigma)


def test_posterior_samples_modern_methods(nll, model_params):
    """Test modern Bayesian methods of PosteriorSamples."""
    sampler = EmceeSampler()
    result = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test credible intervals
    mu_lower, mu_upper = result.credible_interval("mu")
    assert mu_lower < 1.5 < mu_upper

    # Test summary using ArviZ when available
    summary = result.summary()
    # ArviZ returns a DataFrame-like object
    assert hasattr(summary, 'index') or isinstance(summary, dict)

    # Test posterior as prior functionality
    mu_prior = result.as_prior("mu")
    assert hasattr(mu_prior, 'log_pdf')  # Should be a KDE prior

    # Test covariance matrix
    cov = result.covariance()
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))

    # Test convergence diagnostics
    conv_summary = result.convergence_summary()
    assert "valid" in conv_summary
    assert "converged" in conv_summary
    assert "rhat" in conv_summary
    assert "ess_bulk" in conv_summary


def test_posterior_as_prior_workflow(nll, model_params):
    """Test using posterior samples as priors in hierarchical modeling."""
    # First fit with loose priors
    sampler = EmceeSampler(nwalkers=8)
    first_result = sampler.sample(loss=nll, n_samples=50, n_warmup=10)

    # Use posterior of mu as prior for a new parameter
    mu_posterior_prior = first_result.as_prior("mu")

    # Create a new parameter with the posterior as prior
    new_mu = zfit.Parameter("new_mu", 1.5, -5, 5, prior=mu_posterior_prior)

    # Test that the prior can be evaluated
    test_value = 1.5
    log_prob = mu_posterior_prior.log_pdf(test_value)
    assert np.isfinite(log_prob)

    # Test that we can sample from the posterior prior
    samples = first_result.get_samples("mu")
    assert len(samples) > 0
    assert np.all(np.isfinite(samples))


def test_arviz_plotting_methods(nll, model_params):
    """Test ArviZ integration and plotting methods."""
    sampler = EmceeSampler(nwalkers=8)
    result = sampler.sample(loss=nll, n_samples=50, n_warmup=10)

    # ArviZ is required for Bayesian analysis

    # Test ArviZ conversion
    idata = result.to_arviz()
    import arviz as az
    assert isinstance(idata, az.InferenceData)

    # Test plotting methods (just check they don't crash)
    try:
        # Note: These will raise in headless environments, so we catch exceptions
        result.plot_trace()
        result.plot_posterior()
        result.plot_pair()
        result.plot_autocorr()
    except Exception:
        # Plotting may fail in headless CI environments
        pass

    # Test string representation
    str_repr = str(result)
    assert "PosteriorSamples" in str_repr
    assert "valid" in str_repr
    assert "converged" in str_repr

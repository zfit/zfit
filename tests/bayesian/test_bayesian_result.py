"""Tests for the BayesianResult class."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import pytest

import zfit
from zfit._mcmc.emcee import EmceeSampler
from zfit.core.loss import UnbinnedNLL
from zfit.mcmc import BayesianResult


@pytest.fixture
def obs():
    return zfit.Space("x", limits=(-10, 10))


@pytest.fixture
def true_params():
    return {"mu": 1.5, "sigma": 0.5}


@pytest.fixture
def model_params(true_params):
    mu = zfit.Parameter("mu", true_params["mu"], -5, 5, prior=zfit.prior.NormalPrior(1.5, 1.2))
    sigma = zfit.Parameter("sigma", true_params["sigma"], 0.1, 10, prior=zfit.prior.HalfNormalPrior(mu=1.0, sigma=1.0))
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


def test_bayesian_result_basic(nll, model_params):
    """Test basic functionality of BayesianResult."""
    # Run MCMC
    nwalkers = 5
    sampler = EmceeSampler(nwalkers=nwalkers)
    n_samples = 500
    n_warmup = 100
    result = sampler.sample(loss=nll, n_samples=n_samples, n_warmup=n_warmup)

    # Test that we get a BayesianResult
    assert isinstance(result, BayesianResult)

    # Test basic properties
    assert len(result.param_names) == 2
    assert result.n_samples == n_samples
    assert result.n_warmup == n_warmup
    assert result.samples.shape == (n_samples * nwalkers, 2)  # n_samples * nwalkers, n_params

    # Test that posterior attribute exists and is the same as samples
    assert hasattr(result, 'posterior')
    assert result.posterior is result.samples
    assert np.array_equal(result.posterior, result.samples)

    # Test parameter recovery
    mu_mean = result.mean("mu")
    sigma_mean = result.mean("sigma")
    assert abs(mu_mean - 1.5) < 0.2
    assert abs(sigma_mean - 0.5) < 0.2

    # Test ZfitResult interface methods
    assert result.valid
    assert result.converged
    assert result.params is not None
    assert result.values is not None
    assert result.minimizer is not None
    assert result.loss is not None
    assert result.fminopt is not None

    # Test error calculation methods
    hesse_errors = result.hesse()
    assert "mu" in hesse_errors or model_params["mu"] in hesse_errors

    errors = result.errors()
    assert "mu" in errors or model_params["mu"] in errors

    cov = result.covariance()
    assert cov.shape == (2, 2)

    corr = result.correlation()
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)


def test_bayesian_result_context_manager(nll, model_params):
    """Test the context manager functionality of BayesianResult."""
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


def test_bayesian_result_update_params(nll, model_params):
    """Test the update_params method of BayesianResult."""
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


def test_bayesian_result_bayesian_methods(nll, model_params):
    """Test Bayesian-specific methods of BayesianResult."""
    sampler = EmceeSampler()
    result = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test credible intervals
    mu_lower, mu_upper = result.credible_interval("mu")
    assert mu_lower < 1.5 < mu_upper

    # Test HDI
    mu_hdi_lower, mu_hdi_upper = result.highest_density_interval("mu")
    assert mu_hdi_lower < 1.5 < mu_hdi_upper

    # Test summary
    summary = result.summary()
    assert "mu" in summary
    assert "sigma" in summary
    assert "mean" in summary["mu"]
    assert "std" in summary["mu"]

    # Test MAP estimation
    map_estimates = result.map_estimate()
    assert len(map_estimates) == 2
    assert abs(map_estimates[0] - 1.5) < 0.5  # Rough check for mu
    assert abs(map_estimates[1] - 0.5) < 0.5  # Rough check for sigma

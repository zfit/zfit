"""
Shared fixtures for Bayesian tests following pure pytest patterns.

This module provides compositional fixtures that can be combined to create
test scenarios. All fixtures are designed to be small, focused, and reusable.
"""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit.loss import ExtendedUnbinnedNLL, UnbinnedNLL

# =====================================================================
# Global Configuration
# =====================================================================

DEFAULT_SEED = 42


@pytest.fixture(scope="session", autouse=True)
def global_seed():
    """Set global random seed for all tests to ensure reproducibility."""
    import random

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    tf.random.set_seed(DEFAULT_SEED)
    zfit.settings.set_seed(DEFAULT_SEED)

    return DEFAULT_SEED


@pytest.fixture(scope="session", autouse=True)
def disable_numeric_checks():
    """Disable numeric checks for all Bayesian tests."""
    original = zfit.run.numeric_checks
    zfit.run.numeric_checks = False
    yield
    zfit.run.numeric_checks = original


# =====================================================================
# Basic Components
# =====================================================================


@pytest.fixture
def rng():
    """Provide a seeded random number generator."""
    return np.random.RandomState(DEFAULT_SEED)


@pytest.fixture
def make_space():
    """Factory for creating observable spaces."""

    def _make_space(name="x", lower=-5, upper=5):
        return zfit.Space(name, lower=lower, upper=upper)

    return _make_space


@pytest.fixture
def make_parameter():
    """Factory for creating parameters with optional priors."""

    def _make_parameter(name, value, lower=None, upper=None, prior=None):
        return zfit.Parameter(name, value, lower=lower, upper=upper, prior=prior)

    return _make_parameter


# =====================================================================
# Prior Factories
# =====================================================================


@pytest.fixture
def make_normal_prior():
    """Factory for Normal priors."""

    def _make(mu=0.0, sigma=1.0):
        return zfit.prior.Normal(mu=mu, sigma=sigma)

    return _make


@pytest.fixture
def make_uniform_prior():
    """Factory for Uniform priors."""

    def _make(lower=None, upper=None):
        if lower is None and upper is None:
            return zfit.prior.Uniform()
        return zfit.prior.Uniform(lower=lower, upper=upper)

    return _make


@pytest.fixture
def make_halfnormal_prior():
    """Factory for HalfNormal priors."""

    def _make(sigma=1.0, mu=0.0):
        return zfit.prior.HalfNormal(sigma=sigma, mu=mu)

    return _make


@pytest.fixture
def make_gamma_prior():
    """Factory for Gamma priors."""

    def _make(alpha=2.0, beta=1.0, mu=0.0):
        return zfit.prior.Gamma(alpha=alpha, beta=beta, mu=mu)

    return _make


@pytest.fixture
def make_beta_prior():
    """Factory for Beta priors."""

    def _make(alpha=2.0, beta=2.0):
        return zfit.prior.Beta(alpha=alpha, beta=beta)

    return _make


@pytest.fixture
def make_exponential_prior():
    """Factory for Exponential priors."""

    def _make(rate=1.0):
        return zfit.prior.Exponential(rate=rate)

    return _make


@pytest.fixture
def all_prior_factories(
    make_normal_prior,
    make_uniform_prior,
    make_halfnormal_prior,
    make_gamma_prior,
    make_beta_prior,
    make_exponential_prior,
):
    """Dictionary of all prior factories."""
    return {
        "normal": make_normal_prior,
        "uniform": make_uniform_prior,
        "halfnormal": make_halfnormal_prior,
        "gamma": make_gamma_prior,
        "beta": make_beta_prior,
        "exponential": make_exponential_prior,
    }


# =====================================================================
# Model Components
# =====================================================================


@pytest.fixture
def make_gaussian_pdf():
    """Factory for creating Gaussian PDFs."""

    def _make(obs, mu, sigma, extended=None):
        return zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=extended)

    return _make


@pytest.fixture
def make_exponential_pdf():
    """Factory for creating Exponential PDFs."""

    def _make(obs, lam, extended=None):
        return zfit.pdf.Exponential(obs=obs, lam=lam, extended=extended)

    return _make


@pytest.fixture
def make_data():
    """Factory for creating data from arrays or sampling."""

    def _make(obs, array=None, n_events=None, model=None):
        if array is not None:
            if array.ndim == 1:
                array = array[:, np.newaxis]
            return zfit.Data.from_numpy(obs=obs, array=array)
        elif model is not None and n_events is not None:
            return model.sample(n=n_events)
        else:
            msg = "Either provide array or both model and n_events"
            raise ValueError(msg)

    return _make


@pytest.fixture
def make_loss():
    """Factory for creating loss functions."""

    def _make(model, data, extended=False):
        if extended:
            return ExtendedUnbinnedNLL(model=model, data=data)
        return UnbinnedNLL(model=model, data=data)

    return _make


# =====================================================================
# Complete Model Builders
# =====================================================================


@pytest.fixture
def build_simple_gaussian(
    make_space, make_parameter, make_normal_prior, make_halfnormal_prior, make_gaussian_pdf, make_data, make_loss
):
    """Build a complete simple Gaussian model setup."""

    def _build(mu_true=0.0, sigma_true=1.0, n_events=1000, extended=False):
        # Observable space
        obs = make_space("x", -5, 5)

        # Parameters with priors
        mu = make_parameter("mu", mu_true, -2, 2, make_normal_prior(0.0, 2.0))
        sigma = make_parameter("sigma", sigma_true, 0.1, 3.0, make_halfnormal_prior(1.0))

        # Model
        if extended:
            n_sig = make_parameter("n_sig", n_events, 0, 2 * n_events, make_normal_prior(n_events, np.sqrt(n_events)))
            model = make_gaussian_pdf(obs, mu, sigma, extended=n_sig)
            params = [mu, sigma, n_sig]
        else:
            model = make_gaussian_pdf(obs, mu, sigma)
            params = [mu, sigma]

        # Data
        true_data = np.random.normal(mu_true, sigma_true, n_events)
        data = make_data(obs, array=true_data)

        # Loss
        loss = make_loss(model, data, extended=extended)

        return {
            "obs": obs,
            "params": params,
            "model": model,
            "data": data,
            "loss": loss,
            "mu": mu,
            "sigma": sigma,
            "n_sig": n_sig if extended else None,
        }

    return _build


@pytest.fixture
def build_physics_model(
    make_space,
    make_parameter,
    make_normal_prior,
    make_halfnormal_prior,
    make_gaussian_pdf,
    make_exponential_pdf,
    make_data,
    make_loss,
):
    """Build a physics-inspired model with signal and background."""

    def _build(n_sig1=500, n_sig2=300, n_bkg=200):
        obs = make_space("x", 0, 10)

        # Background
        lambda_bkg = make_parameter("lambda_bkg", -0.5, -2.0, 0.0, make_normal_prior(-0.5, 0.3))

        # Signal 1
        mean1 = make_parameter("mean1", 3.0, 2.5, 3.5, make_normal_prior(3.0, 0.2))
        sigma1 = make_parameter("sigma1", 0.3, 0.1, 0.6, make_halfnormal_prior(0.2))

        # Signal 2
        mean2 = make_parameter("mean2", 6.0, 5.5, 6.5, make_normal_prior(6.0, 0.2))
        sigma2 = make_parameter("sigma2", 0.4, 0.1, 0.8, make_halfnormal_prior(0.2))

        # Yields
        yield_sig1 = make_parameter("yield_sig1", n_sig1, 0, 1500, make_normal_prior(n_sig1, 100))
        yield_sig2 = make_parameter("yield_sig2", n_sig2, 0, 1000, make_normal_prior(n_sig2, 100))
        yield_bkg = make_parameter("yield_bkg", n_bkg, 0, 800, make_normal_prior(n_bkg, 50))

        # PDFs
        bkg = make_exponential_pdf(obs, lambda_bkg, extended=yield_bkg)
        peak1 = make_gaussian_pdf(obs, mean1, sigma1, extended=yield_sig1)
        peak2 = make_gaussian_pdf(obs, mean2, sigma2, extended=yield_sig2)

        # Combined model
        model = zfit.pdf.SumPDF([peak1, peak2, bkg])
        params = model.get_params()

        # Data
        n_total = n_sig1 + n_sig2 + n_bkg
        data = model.sample(n=n_total)

        # Loss
        loss = make_loss(model, data, extended=True)

        return {
            "obs": obs,
            "model": model,
            "params": params,
            "data": data,
            "loss": loss,
            "components": {"bkg": bkg, "peak1": peak1, "peak2": peak2},
            "yields": {"sig1": yield_sig1, "sig2": yield_sig2, "bkg": yield_bkg},
        }

    return _build


# =====================================================================
# Sampler Configuration
# =====================================================================


@pytest.fixture(params=["quick", "medium", "full"])
def sampling_config(request):
    """Parameterized fixture providing different sampling configurations."""
    configs = {
        "quick": {"nwalkers": 8, "n_samples": 50, "n_warmup": 20},
        "medium": {"nwalkers": 16, "n_samples": 200, "n_warmup": 100},
        "full": {"nwalkers": 24, "n_samples": 500, "n_warmup": 200},
    }
    return configs[request.param]


@pytest.fixture
def make_sampler():
    """Factory for creating EmceeSampler instances."""

    def _make(nwalkers=16, verbosity=0, backend=None):
        return zfit.mcmc.EmceeSampler(nwalkers=nwalkers, verbosity=verbosity, backend=backend)

    return _make


@pytest.fixture
def sample_posterior(make_sampler):
    """Function to perform MCMC sampling."""

    def _sample(loss, params, nwalkers=16, n_samples=100, n_warmup=50):
        sampler = make_sampler(nwalkers=nwalkers)
        return sampler.sample(loss=loss, params=params, n_samples=n_samples, n_warmup=n_warmup)

    return _sample


# =====================================================================
# Validation Helpers
# =====================================================================


@pytest.fixture
def assert_samples_valid():
    """Assertion helper for validating samples."""

    def _assert(samples, expected_shape=None, check_finite=True, check_positive=False):
        samples_array = np.asarray(samples)

        if expected_shape is not None:
            assert samples_array.shape == expected_shape

        if check_finite:
            assert np.all(np.isfinite(samples_array)), "Samples contain non-finite values"

        if check_positive:
            assert np.all(samples_array >= 0), "Samples contain negative values"

        return samples_array

    return _assert


@pytest.fixture
def assert_parameter_recovered():
    """Assertion helper for parameter recovery."""

    def _assert(samples, true_value, tolerance=0.5, use_median=False):
        samples_array = np.asarray(samples).flatten()
        estimate = np.median(samples_array) if use_median else np.mean(samples_array)

        assert abs(estimate - true_value) < tolerance, (
            f"Parameter not recovered: estimate={estimate:.3f}, true={true_value:.3f}"
        )

        return estimate

    return _assert


@pytest.fixture
def assert_posterior_valid():
    """Assertion helper for posterior validation."""

    def _assert(posterior, expected_params=None, check_convergence=False):
        # Basic validity
        assert posterior is not None
        assert hasattr(posterior, "samples")
        assert len(posterior.samples) > 0

        # Parameter checks
        if expected_params is not None:
            assert len(posterior.param_names) == len(expected_params)
            for param_name in expected_params:
                assert param_name in posterior.param_names

        # Convergence checks
        if check_convergence:
            assert posterior.valid, "Posterior is not valid"
            if hasattr(posterior, "converged"):
                assert posterior.converged, "Posterior has not converged"

        return posterior

    return _assert


# =====================================================================
# High-Statistics Posteriors (Session-scoped for expensive operations)
# =====================================================================


@pytest.fixture(scope="session")
def gaussian_posterior_highstats(global_seed):
    """Pre-computed high-statistics posterior for a simple Gaussian model."""
    # Build model
    obs = zfit.Space("x", lower=-5, upper=5)
    mu = zfit.Parameter("mu", 0.0, prior=zfit.prior.Normal(mu=0.0, sigma=2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10.0, prior=zfit.prior.HalfNormal(sigma=1.0))

    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    data = zfit.Data.from_numpy(obs=obs, array=np.random.normal(0, 1, 10000)[:, np.newaxis])
    loss = UnbinnedNLL(model=model, data=data)

    # Sample
    sampler = zfit.mcmc.EmceeSampler(nwalkers=24, verbosity=0)
    posterior = sampler.sample(loss=loss, params=[mu, sigma], n_samples=500, n_warmup=200)

    return {
        "posterior": posterior,
        "model": model,
        "params": [mu, sigma],
        "true_values": {"mu": 0.0, "sigma": 1.0},
        "mu": mu,
        "sigma": sigma,
        "loss": loss,
        "data": data,
        "obs": obs,
    }


@pytest.fixture(scope="session")
def physics_posterior_highstats(global_seed):
    """Pre-computed high-statistics posterior for a physics model."""
    # This is expensive, so we compute it once

    # We need to manually build since we can't use function-scoped fixtures
    obs = zfit.Space("x", lower=0, upper=10)

    # Create all parameters with priors
    lambda_bkg = zfit.Parameter("lambda_bkg", -0.5, -2.0, 0.0, prior=zfit.prior.Normal(mu=-0.5, sigma=0.3))
    mean1 = zfit.Parameter("mean1", 3.0, 2.5, 3.5, prior=zfit.prior.Normal(mu=3.0, sigma=0.2))
    sigma1 = zfit.Parameter("sigma1", 0.3, 0.1, 0.6, prior=zfit.prior.HalfNormal(sigma=0.2))
    mean2 = zfit.Parameter("mean2", 6.0, 5.5, 6.5, prior=zfit.prior.Normal(mu=6.0, sigma=0.2))
    sigma2 = zfit.Parameter("sigma2", 0.4, 0.1, 0.8, prior=zfit.prior.HalfNormal(sigma=0.2))

    yield_sig1 = zfit.Parameter("yield_sig1", 500, 0, 1500, prior=zfit.prior.Normal(mu=500, sigma=100))
    yield_sig2 = zfit.Parameter("yield_sig2", 300, 0, 1000, prior=zfit.prior.Normal(mu=300, sigma=100))
    yield_bkg = zfit.Parameter("yield_bkg", 200, 0, 800, prior=zfit.prior.Normal(mu=200, sigma=50))

    # Build model
    bkg = zfit.pdf.Exponential(obs=obs, lam=lambda_bkg, extended=yield_bkg)
    peak1 = zfit.pdf.Gauss(obs=obs, mu=mean1, sigma=sigma1, extended=yield_sig1)
    peak2 = zfit.pdf.Gauss(obs=obs, mu=mean2, sigma=sigma2, extended=yield_sig2)

    model = zfit.pdf.SumPDF([peak1, peak2, bkg])
    params = model.get_params()

    data = model.sample(n=1000)
    loss = UnbinnedNLL(model=model, data=data)

    # Sample
    sampler = zfit.mcmc.EmceeSampler(nwalkers=20, verbosity=0)
    posterior = sampler.sample(loss=loss, params=params, n_samples=300, n_warmup=150)

    return {
        "posterior": posterior,
        "model": model,
        "params": params,
        "components": {"bkg": bkg, "peak1": peak1, "peak2": peak2},
        "loss": loss,
        "data": data,
        "obs": obs,
    }


# =====================================================================
# Aliases for compatibility with existing tests
# =====================================================================


@pytest.fixture(scope="session")
def simple_gaussian_high_stats_posterior(gaussian_posterior_highstats):
    """Alias for gaussian_posterior_highstats for compatibility."""
    return gaussian_posterior_highstats


@pytest.fixture(scope="session")
def physics_model_high_stats_posterior(physics_posterior_highstats):
    """Alias for physics_posterior_highstats for compatibility."""
    # Add additional fields that tests might expect
    result = physics_posterior_highstats.copy()
    # Create loss if not present
    if "loss" not in result:
        from zfit.loss import UnbinnedNLL

        result["loss"] = UnbinnedNLL(model=result["model"], data=result["model"].sample(n=1000))
    # Add data if not present
    if "data" not in result:
        result["data"] = result["model"].sample(n=1000)
    return result


# =====================================================================
# Test Markers
# =====================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "sampling: marks tests that perform MCMC sampling")
    config.addinivalue_line("markers", "flaky: marks tests that may occasionally fail due to randomness")

"""Tests for vectorized credible_interval implementation."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit._mcmc.emcee import EmceeSampler
from zfit.core.loss import UnbinnedNLL


@pytest.fixture
def simple_posterior():
    """Create a simple posterior for testing."""
    # Create simple model
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 0.0, -5, 5, prior=zfit.prior.Normal(0.0, 1.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=zfit.prior.HalfNormal(sigma=1.0))
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Generate data
    data = np.random.normal(0.0, 1.0, 500)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)

    # Create loss and sample
    loss = UnbinnedNLL(model=model, data=dataset)
    sampler = EmceeSampler(nwalkers=8)
    posterior = sampler.sample(loss=loss, n_samples=100, n_warmup=50)

    return posterior, [mu, sigma]


def test_credible_interval_single_param(simple_posterior):
    """Test credible interval for single parameter."""
    posterior, params = simple_posterior
    mu = params[0]

    # Test single parameter
    lower, upper = posterior.credible_interval(mu, alpha=0.05)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper

    # Test with parameter name
    lower2, upper2 = posterior.credible_interval("mu", alpha=0.05)
    assert lower == lower2
    assert upper == upper2


def test_credible_interval_multiple_params(simple_posterior):
    """Test credible interval for multiple parameters."""
    posterior, params = simple_posterior

    # Test with list of parameters
    lowers, uppers = posterior.credible_interval(params, alpha=0.05)
    assert isinstance(lowers, np.ndarray)
    assert isinstance(uppers, np.ndarray)
    assert len(lowers) == 2
    assert len(uppers) == 2
    assert all(lowers < uppers)

    # Test with parameter names
    lowers2, uppers2 = posterior.credible_interval(["mu", "sigma"], alpha=0.05)
    np.testing.assert_array_almost_equal(lowers, lowers2)
    np.testing.assert_array_almost_equal(uppers, uppers2)


def test_credible_interval_all_params(simple_posterior):
    """Test credible interval for all parameters."""
    posterior, params = simple_posterior

    # Test with None (all parameters)
    lowers, uppers = posterior.credible_interval(None, alpha=0.05)
    assert isinstance(lowers, np.ndarray)
    assert isinstance(uppers, np.ndarray)
    assert len(lowers) == 2
    assert len(uppers) == 2
    assert all(lowers < uppers)


def test_credible_interval_different_alphas(simple_posterior):
    """Test credible intervals with different significance levels."""
    posterior, params = simple_posterior
    mu = params[0]

    # 68% interval (1 sigma)
    lower_68, upper_68 = posterior.credible_interval(mu, alpha=0.32)

    # 95% interval (2 sigma)
    lower_95, upper_95 = posterior.credible_interval(mu, alpha=0.05)

    # 99% interval (~3 sigma)
    lower_99, upper_99 = posterior.credible_interval(mu, alpha=0.01)

    # Check that intervals are nested
    assert lower_99 < lower_95 < lower_68
    assert upper_68 < upper_95 < upper_99


def test_credible_interval_sigma_param(simple_posterior):
    """Test credible interval using sigma parameter."""
    posterior, params = simple_posterior
    mu = params[0]

    # 1 sigma interval (~68%)
    lower_1s, upper_1s = posterior.credible_interval(mu, sigma=1)

    # 2 sigma interval (~95%)
    lower_2s, upper_2s = posterior.credible_interval(mu, sigma=2)

    # Check that intervals are nested
    assert lower_2s < lower_1s
    assert upper_1s < upper_2s

    # Check approximate correspondence with alpha
    lower_95, upper_95 = posterior.credible_interval(mu, alpha=0.05)
    assert abs(lower_2s - lower_95) < 0.1  # Should be close
    assert abs(upper_2s - upper_95) < 0.1


def test_credible_interval_mixed_param_types(simple_posterior):
    """Test credible interval with mixed parameter types."""
    posterior, params = simple_posterior
    mu = params[0]

    # Mix of parameter object and name
    lowers, uppers = posterior.credible_interval([mu, "sigma"], alpha=0.05)
    assert len(lowers) == 2
    assert len(uppers) == 2
    assert all(lowers < uppers)


def test_credible_interval_performance():
    """Test that vectorized implementation is actually faster."""
    # Create a posterior with more parameters
    obs = zfit.Space("x", limits=(-10, 10))
    params = []
    for i in range(10):
        p = zfit.Parameter(f"p{i}", 0.0, -5, 5, prior=zfit.prior.Normal(0.0, 1.0))
        params.append(p)

    # Create fake samples for speed test
    n_samples = 10000
    n_params = len(params)
    samples = np.random.normal(0, 1, (n_samples, n_params))

    # Create mock posterior
    from zfit.mcmc import PosteriorSamples
    posterior = PosteriorSamples(
        samples=samples,
        params=params,
        loss=None,
        sampler=None,
        n_warmup=0,
        n_samples=n_samples
    )

    # Test vectorized version (should be fast)
    import time
    start = time.time()
    lowers, uppers = posterior.credible_interval(params, alpha=0.05)
    vectorized_time = time.time() - start

    # The vectorized version should handle 10 parameters efficiently
    assert len(lowers) == 10
    assert len(uppers) == 10
    assert vectorized_time < 0.1  # Should be very fast


def test_credible_interval_edge_cases(simple_posterior):
    """Test edge cases for credible interval."""
    posterior, params = simple_posterior

    # Test with empty list
    with pytest.raises((ValueError, IndexError)):
        posterior.credible_interval([])

    # Test with invalid parameter name
    with pytest.raises((ValueError, KeyError)):
        posterior.credible_interval("nonexistent_param")

    # Test with very small alpha (should give wide interval)
    lower, upper = posterior.credible_interval(params[0], alpha=0.001)
    assert upper - lower > 0  # Should have non-zero width

    # Test with very large alpha (should give narrow interval)
    lower, upper = posterior.credible_interval(params[0], alpha=0.999)
    assert upper - lower < 1.0  # Should be narrow


if __name__ == "__main__":
    pytest.main([__file__])

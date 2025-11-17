"""Tests for refactored PosteriorSamples methods."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.mcmc import PosteriorSamples


@pytest.fixture
def mock_posterior():
    """Create a mock posterior for testing."""
    # Create fake parameters
    params = [
        zfit.Parameter("a", 1.0),
        zfit.Parameter("b", 2.0),
        zfit.Parameter("c", 3.0),
    ]

    # Create fake samples
    n_samples = 1000
    samples = np.random.normal(0, 1, (n_samples, 3))
    samples[:, 0] += 1  # Center around 1 for param a
    samples[:, 1] += 2  # Center around 2 for param b
    samples[:, 2] += 3  # Center around 3 for param c

    return PosteriorSamples(
        samples=samples,
        params=params,
        loss=None,
        sampler=None,
        n_warmup=0,
        n_samples=n_samples
    )


def test_get_param_positions_single(mock_posterior):
    """Test _get_param_positions with single parameters."""
    posterior = mock_posterior

    # Test with string
    positions = posterior._get_param_positions("a")
    assert positions == [0]

    # Test with parameter object
    positions = posterior._get_param_positions(posterior.params[1])
    assert positions == [1]


def test_get_param_positions_multiple(mock_posterior):
    """Test _get_param_positions with multiple parameters."""
    posterior = mock_posterior

    # Test with list of strings
    positions = posterior._get_param_positions(["a", "c"])
    assert positions == [0, 2]

    # Test with mixed types (strings and parameter objects)
    positions = posterior._get_param_positions(["a", posterior.params[1]])
    assert positions == [0, 1]

    # Test with tuple
    positions = posterior._get_param_positions(("b", "c"))
    assert positions == [1, 2]


def test_mean_vectorized(mock_posterior):
    """Test vectorized mean method."""
    posterior = mock_posterior

    # Test single parameter
    mean_a = posterior.mean("a")
    assert isinstance(mean_a, float)
    assert 0.5 < mean_a < 1.5  # Should be around 1

    # Test multiple parameters
    means = posterior.mean(["a", "b"])
    assert isinstance(means, np.ndarray)
    assert len(means) == 2
    assert 0.5 < means[0] < 1.5  # Should be around 1
    assert 1.5 < means[1] < 2.5  # Should be around 2

    # Test all parameters
    all_means = posterior.mean()
    assert len(all_means) == 3


def test_std_vectorized(mock_posterior):
    """Test vectorized std method."""
    posterior = mock_posterior

    # Test single parameter
    std_a = posterior.std("a")
    assert isinstance(std_a, float)
    assert 0.8 < std_a < 1.2  # Should be around 1

    # Test multiple parameters
    stds = posterior.std(["a", "b", "c"])
    assert isinstance(stds, np.ndarray)
    assert len(stds) == 3
    assert all(0.8 < s < 1.2 for s in stds)  # Should all be around 1


def test_get_samples_vectorized(mock_posterior):
    """Test vectorized get_samples method."""
    posterior = mock_posterior

    # Test single parameter (should return 1D array)
    samples_a = posterior.get_samples("a")
    assert samples_a.ndim == 1
    assert len(samples_a) == 1000

    # Test multiple parameters (should return 2D array)
    samples_ab = posterior.get_samples(["a", "b"])
    assert samples_ab.ndim == 2
    assert samples_ab.shape == (1000, 2)

    # Test all parameters
    all_samples = posterior.get_samples()
    assert all_samples.shape == (1000, 3)


def test_covariance_vectorized(mock_posterior):
    """Test vectorized covariance method."""
    posterior = mock_posterior

    # Test specific parameters
    cov_ab = posterior.covariance(["a", "b"])
    assert cov_ab.shape == (2, 2)
    assert np.all(np.isfinite(cov_ab))

    # Test all parameters
    cov_all = posterior.covariance()
    assert cov_all.shape == (3, 3)

    # Covariance matrix should be symmetric
    assert np.allclose(cov_all, cov_all.T)


def test_credible_interval_consistency(mock_posterior):
    """Test that credible interval works consistently with refactoring."""
    posterior = mock_posterior

    # Test that single parameter returns scalars
    lower, upper = posterior.credible_interval("a")
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper

    # Test that multiple parameters return arrays
    lowers, uppers = posterior.credible_interval(["a", "b"])
    assert isinstance(lowers, np.ndarray)
    assert isinstance(uppers, np.ndarray)
    assert len(lowers) == 2
    assert all(lowers < uppers)


def test_update_params_subset(mock_posterior):
    """Test updating a subset of parameters."""
    posterior = mock_posterior
    params = posterior.params

    # Store original values
    orig_values = [p.value() for p in params]

    # Update only parameters a and c
    posterior.update_params([params[0], params[2]])

    # Check that a and c were updated to their means
    assert params[0].value() != orig_values[0]
    assert params[1].value() == orig_values[1]  # b should not change
    assert params[2].value() != orig_values[2]

    # Restore original values
    for p, val in zip(params, orig_values):
        p.set_value(val)


def test_error_handling(mock_posterior):
    """Test error handling in _get_param_positions."""
    posterior = mock_posterior

    # Test with invalid parameter name
    with pytest.raises(ValueError, match="Parameter 'invalid' not found"):
        posterior._get_param_positions("invalid")

    # Test with integer (no longer supported)
    with pytest.raises(TypeError, match="Invalid parameter type.*Expected string.*or ZfitParameter"):
        posterior._get_param_positions(10)

    # Test with invalid type
    with pytest.raises(TypeError, match="Invalid parameter type"):
        posterior._get_param_positions({"invalid": "type"})

    # Test empty list
    positions = posterior._get_param_positions([])
    assert positions == []


if __name__ == "__main__":
    pytest.main([__file__])

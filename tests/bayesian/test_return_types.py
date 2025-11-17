"""Test return types for posterior methods based on input container type."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit._bayesian.posterior import PosteriorSamples


@pytest.fixture
def mock_posterior():
    """Create a mock posterior with 2 parameters for testing."""
    # Create mock parameters
    mu = zfit.Parameter("mu", 0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0.1, 3)
    params = [mu, sigma]

    # Generate mock samples (100 samples, 2 parameters)
    np.random.seed(42)
    samples = np.random.normal([0, 1], [0.1, 0.2], (100, 2))

    # Create mock objects for required arguments
    class MockLoss:
        def __str__(self):
            return "MockLoss"

    class MockSampler:
        def __str__(self):
            return "MockSampler"
        nwalkers = 4

    # Create PosteriorSamples instance
    posterior = PosteriorSamples(
        samples=samples,
        params=params,
        loss=MockLoss(),
        sampler=MockSampler(),
        n_warmup=10,
        n_samples=25
    )

    return posterior, mu, sigma


@pytest.fixture
def params_input_cases(mock_posterior):
    """Create test cases for different parameter input formats."""
    posterior, mu, sigma = mock_posterior

    return [
        # (input, description, expect_scalar_1d, expected_length)
        ("mu", "single_string", True, 1),
        (mu, "single_param_obj", True, 1),
        (["mu"], "list_single_string", False, 1),
        ([mu], "list_single_param_obj", False, 1),
        (["mu", "sigma"], "list_multiple_strings", False, 2),
        ([mu, sigma], "list_multiple_param_objs", False, 2),
        (("mu",), "tuple_single_string", False, 1),
        ((mu,), "tuple_single_param_obj", False, 1),
        (("mu", "sigma"), "tuple_multiple_strings", False, 2),
        ((mu, sigma), "tuple_multiple_param_objs", False, 2),
        (None, "none_all_params", False, 2),
    ]


def test_get_samples_single_parameter(mock_posterior):
    """Test get_samples with single parameter returns 1D array."""
    posterior, mu, _ = mock_posterior

    samples_single = posterior.get_samples("mu")
    assert samples_single.ndim == 1
    assert len(samples_single) == 100

    samples_single_obj = posterior.get_samples(mu)
    assert samples_single_obj.ndim == 1
    assert len(samples_single_obj) == 100


def test_get_samples_collection_parameters(mock_posterior):
    """Test get_samples with collection of parameters returns 2D array."""
    posterior, mu, sigma = mock_posterior

    # Single parameter in list -> 2D array
    samples_list_single = posterior.get_samples(["mu"])
    assert samples_list_single.ndim == 2
    assert samples_list_single.shape == (100, 1)

    # Multiple parameters -> 2D array
    samples_multiple = posterior.get_samples(["mu", "sigma"])
    assert samples_multiple.ndim == 2
    assert samples_multiple.shape == (100, 2)

    # All parameters (None) -> 2D array
    samples_all = posterior.get_samples(None)
    assert samples_all.ndim == 2
    assert samples_all.shape == (100, 2)


def test_get_samples_content_consistency(mock_posterior):
    """Test that get_samples content is the same regardless of input format."""
    posterior, mu, _ = mock_posterior

    samples_single = posterior.get_samples("mu")
    samples_list_single = posterior.get_samples(["mu"])
    samples_single_obj = posterior.get_samples(mu)
    samples_list_single_obj = posterior.get_samples([mu])

    np.testing.assert_array_equal(samples_single, samples_list_single[:, 0])
    np.testing.assert_array_equal(samples_single_obj, samples_list_single_obj[:, 0])


@pytest.mark.parametrize("method_name", ["mean", "std"])
def test_scalar_methods_single_parameter(mock_posterior, method_name):
    """Test methods that return scalars for single parameters."""
    posterior, mu, _ = mock_posterior
    method = getattr(posterior, method_name)

    # Single parameter -> scalar
    result_single = method("mu")
    assert np.isscalar(result_single)
    assert isinstance(result_single, float)

    result_single_obj = method(mu)
    assert np.isscalar(result_single_obj)
    assert isinstance(result_single_obj, float)


@pytest.mark.parametrize("method_name", ["mean", "std"])
def test_scalar_methods_collection_parameters(mock_posterior, method_name):
    """Test methods that return arrays for collection of parameters."""
    posterior, mu, sigma = mock_posterior
    method = getattr(posterior, method_name)

    # Single parameter in list -> array
    result_list_single = method(["mu"])
    assert isinstance(result_list_single, np.ndarray)
    assert result_list_single.shape == (1,)

    # Multiple parameters -> array
    result_multiple = method(["mu", "sigma"])
    assert isinstance(result_multiple, np.ndarray)
    assert result_multiple.shape == (2,)

    # All parameters (None) -> array
    result_all = method(None)
    assert isinstance(result_all, np.ndarray)
    assert result_all.shape == (2,)


@pytest.mark.parametrize("method_name", ["mean", "std"])
def test_scalar_methods_content_consistency(mock_posterior, method_name):
    """Test content consistency for scalar methods."""
    posterior, mu, _ = mock_posterior
    method = getattr(posterior, method_name)

    result_single = method("mu")
    result_list_single = method(["mu"])
    result_single_obj = method(mu)
    result_list_single_obj = method([mu])

    assert result_single == result_list_single[0]
    assert result_single_obj == result_list_single_obj[0]


def test_credible_interval_single_parameter(mock_posterior):
    """Test credible_interval with single parameter returns tuple of scalars."""
    posterior, mu, _ = mock_posterior

    lower_single, upper_single = posterior.credible_interval("mu")
    assert np.isscalar(lower_single) and np.isscalar(upper_single)
    assert isinstance(lower_single, float) and isinstance(upper_single, float)

    lower_single_obj, upper_single_obj = posterior.credible_interval(mu)
    assert np.isscalar(lower_single_obj) and np.isscalar(upper_single_obj)
    assert isinstance(lower_single_obj, float) and isinstance(upper_single_obj, float)


def test_credible_interval_collection_parameters(mock_posterior):
    """Test credible_interval with collection returns tuple of arrays."""
    posterior, mu, sigma = mock_posterior

    # Single parameter in list -> tuple of arrays
    lower_list_single, upper_list_single = posterior.credible_interval(["mu"])
    assert isinstance(lower_list_single, np.ndarray) and isinstance(upper_list_single, np.ndarray)
    assert lower_list_single.shape == (1,) and upper_list_single.shape == (1,)

    # Multiple parameters -> tuple of arrays
    lower_multiple, upper_multiple = posterior.credible_interval(["mu", "sigma"])
    assert isinstance(lower_multiple, np.ndarray) and isinstance(upper_multiple, np.ndarray)
    assert lower_multiple.shape == (2,) and upper_multiple.shape == (2,)

    # All parameters (None) -> tuple of arrays
    lower_all, upper_all = posterior.credible_interval(None)
    assert isinstance(lower_all, np.ndarray) and isinstance(upper_all, np.ndarray)
    assert lower_all.shape == (2,) and upper_all.shape == (2,)


def test_credible_interval_content_consistency(mock_posterior):
    """Test credible_interval content consistency."""
    posterior, mu, _ = mock_posterior

    lower_single, upper_single = posterior.credible_interval("mu")
    lower_list_single, upper_list_single = posterior.credible_interval(["mu"])
    lower_single_obj, upper_single_obj = posterior.credible_interval(mu)
    lower_list_single_obj, upper_list_single_obj = posterior.credible_interval([mu])

    assert lower_single == lower_list_single[0] and upper_single == upper_list_single[0]
    assert lower_single_obj == lower_list_single_obj[0] and upper_single_obj == upper_list_single_obj[0]


def test_covariance_single_parameter(mock_posterior):
    """Test covariance with single parameter returns scalar (variance)."""
    posterior, mu, _ = mock_posterior

    cov_single = posterior.covariance("mu")
    assert np.isscalar(cov_single)
    assert isinstance(cov_single, (float, np.floating))

    cov_single_obj = posterior.covariance(mu)
    assert np.isscalar(cov_single_obj)
    assert isinstance(cov_single_obj, (float, np.floating))


def test_covariance_collection_parameters(mock_posterior):
    """Test covariance with collection returns matrix."""
    posterior, mu, sigma = mock_posterior

    # Single parameter in list -> 1x1 matrix
    cov_list_single = posterior.covariance(["mu"])
    assert isinstance(cov_list_single, np.ndarray)
    assert cov_list_single.shape == (1, 1)

    # Multiple parameters -> NxN matrix
    cov_multiple = posterior.covariance(["mu", "sigma"])
    assert isinstance(cov_multiple, np.ndarray)
    assert cov_multiple.shape == (2, 2)

    # All parameters (None) -> NxN matrix
    cov_all = posterior.covariance(None)
    assert isinstance(cov_all, np.ndarray)
    assert cov_all.shape == (2, 2)


def test_covariance_content_consistency(mock_posterior):
    """Test covariance content consistency."""
    posterior, mu, _ = mock_posterior

    cov_single = posterior.covariance("mu")
    cov_list_single = posterior.covariance(["mu"])
    cov_single_obj = posterior.covariance(mu)
    cov_list_single_obj = posterior.covariance([mu])

    # The scalar variance should equal the diagonal element of the 1x1 matrix
    np.testing.assert_allclose(cov_single, cov_list_single[0, 0])
    np.testing.assert_allclose(cov_single_obj, cov_list_single_obj[0, 0])


def test_symerr_inherits_std_behavior(mock_posterior):
    """Test that symerr inherits the return type behavior from std."""
    posterior, mu, _ = mock_posterior

    # Single parameter -> scalar
    symerr_single = posterior.symerr("mu")
    assert np.isscalar(symerr_single)

    # Single parameter in list -> array
    symerr_list_single = posterior.symerr(["mu"])
    assert isinstance(symerr_list_single, np.ndarray)
    assert symerr_list_single.shape == (1,)

    # Verify relationship: symerr = sigma * std (default sigma=1)
    std_single = posterior.std("mu")
    assert symerr_single == std_single

    std_list_single = posterior.std(["mu"])
    np.testing.assert_array_equal(symerr_list_single, std_list_single)


def test_tuple_input_behavior(mock_posterior):
    """Test that tuple inputs are treated as containers."""
    posterior, mu, sigma = mock_posterior

    # Tuple with single parameter -> should return array/2D
    mean_tuple_single = posterior.mean(("mu",))
    assert isinstance(mean_tuple_single, np.ndarray)
    assert mean_tuple_single.shape == (1,)

    samples_tuple_single = posterior.get_samples(("mu",))
    assert samples_tuple_single.ndim == 2
    assert samples_tuple_single.shape == (100, 1)

    # Tuple with multiple parameters -> should return array/2D
    mean_tuple_multiple = posterior.mean(("mu", "sigma"))
    assert isinstance(mean_tuple_multiple, np.ndarray)
    assert mean_tuple_multiple.shape == (2,)

    samples_tuple_multiple = posterior.get_samples(("mu", "sigma"))
    assert samples_tuple_multiple.ndim == 2
    assert samples_tuple_multiple.shape == (100, 2)


@pytest.mark.parametrize("method_name", ["get_samples", "mean", "std", "credible_interval", "covariance"])
def test_empty_list_raises_error(mock_posterior, method_name):
    """Test that empty list raises ValueError for all methods."""
    posterior, _, _ = mock_posterior
    method = getattr(posterior, method_name)

    with pytest.raises(ValueError):
        method([])


@pytest.mark.parametrize("method_name", ["get_samples", "mean", "std", "credible_interval", "covariance"])
def test_invalid_parameter_raises_error(mock_posterior, method_name):
    """Test that invalid parameter name raises ValueError for all methods."""
    posterior, _, _ = mock_posterior
    method = getattr(posterior, method_name)

    with pytest.raises(ValueError):
        method("nonexistent")


def test_consistency_across_methods(mock_posterior, params_input_cases):
    """Test that all methods treat input types consistently."""
    posterior, _, _ = mock_posterior

    for input_val, description, expect_scalar_1d, expected_length in params_input_cases:
        # Test get_samples
        samples = posterior.get_samples(input_val)
        if expect_scalar_1d:
            assert samples.ndim == 1, f"get_samples({description}) should return 1D"
            assert len(samples) == 100, f"get_samples({description}) wrong length"
        else:
            assert samples.ndim == 2, f"get_samples({description}) should return 2D"
            assert samples.shape == (100, expected_length), f"get_samples({description}) wrong shape"

        # Test mean
        mean_val = posterior.mean(input_val)
        if expect_scalar_1d:
            assert np.isscalar(mean_val), f"mean({description}) should return scalar"
        else:
            assert isinstance(mean_val, np.ndarray), f"mean({description}) should return array"
            assert mean_val.shape == (expected_length,), f"mean({description}) wrong shape"

        # Test std
        std_val = posterior.std(input_val)
        if expect_scalar_1d:
            assert np.isscalar(std_val), f"std({description}) should return scalar"
        else:
            assert isinstance(std_val, np.ndarray), f"std({description}) should return array"
            assert std_val.shape == (expected_length,), f"std({description}) wrong shape"

        # Test credible_interval
        lower, upper = posterior.credible_interval(input_val)
        if expect_scalar_1d:
            assert np.isscalar(lower) and np.isscalar(upper), f"credible_interval({description}) should return scalars"
        else:
            assert isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray), f"credible_interval({description}) should return arrays"
            assert lower.shape == (expected_length,) and upper.shape == (expected_length,), f"credible_interval({description}) wrong shape"

        # Test covariance
        cov_val = posterior.covariance(input_val)
        if expect_scalar_1d:
            assert np.isscalar(cov_val), f"covariance({description}) should return scalar"
        else:
            assert isinstance(cov_val, np.ndarray), f"covariance({description}) should return array"
            assert cov_val.shape == (expected_length, expected_length), f"covariance({description}) wrong shape"

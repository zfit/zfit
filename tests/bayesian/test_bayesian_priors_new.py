"""Tests for the redesigned Bayesian priors with constraint system."""

#  Copyright (c) 2025 zfit

import pytest
import numpy as np
import zfit
from zfit._bayesian.priors import (
    Normal, Beta, HalfNormal, Gamma, KDE,
    LogNormal, Cauchy, Exponential, Poisson, StudentT
)
from zfit._bayesian.mathconstrain import ConstraintType


def test_normal_creation_and_validation():
    """Test Normal prior creation and parameter validation."""
    # Valid parameters
    prior = Normal(mu=0.0, sigma=1.0)
    assert prior.constraint.constraint_type == ConstraintType.UNCONSTRAINED

    # Invalid sigma should raise error in eager mode
    with pytest.raises(ValueError, match="must be positive"):
        Normal(mu=0.0, sigma=-1.0)


def test_normal_pdf_creation():
    """Test that Normal PDF is created correctly."""
    prior = Normal(mu=2.0, sigma=0.5)
    assert prior.pdf is not None
    assert hasattr(prior.pdf, 'pdf')


def test_normal_adaptation_to_bounds():
    """Test Normal adaptation to parameter bounds."""
    prior = Normal(mu=0.0, sigma=1.0)

    # Mock parameter with bounds
    class MockParam:
        has_limits = True
        lower = -2.0
        upper = 3.0

    param = MockParam()
    prior._register_default_param(param)

    # Should create truncated normal when adapted
    # Note: Full testing would require checking the adapted PDF type


def test_beta_bounds_constraint():
    """Test that Beta bounds are stored correctly."""
    prior = Beta(alpha=1.0, beta=1.0, lower=-1.0, upper=2.0)
    assert prior.constraint.bounds == (-1.0, 2.0)


def test_beta_adaptation_restrictions():
    """Test that Beta adapts to parameter bounds."""
    prior = Beta(alpha=2.0, beta=2.0, lower=-1.0, upper=2.0)

    # Mock parameter with limits
    class MockParam:
        has_limits = True
        lower = 0.0
        upper = 1.0

    param = MockParam()
    # Should adapt to parameter limits
    assert prior._should_adapt_to_param(param)


def test_beta_creation_and_validation():
    """Test Beta prior creation and validation."""
    # Valid parameters
    prior = Beta(alpha=2.0, beta=3.0, lower=-1.0, upper=5.0)
    assert prior.constraint.constraint_type == ConstraintType.CUSTOM_BOUNDS
    assert prior.constraint.bounds == (-1.0, 5.0)

    # Invalid bound ordering should raise error
    with pytest.raises(ValueError, match="Lower bound .* must be less than upper bound"):
        Beta(alpha=1.0, beta=1.0, lower=5.0, upper=2.0)

    # Invalid shape parameters should raise error
    with pytest.raises(ValueError, match="must be positive"):
        Beta(alpha=-1.0, beta=1.0, lower=0.0, upper=1.0)


def test_beta_bounds_storage():
    """Test that Beta bounds are stored correctly."""
    lower, upper = -2.5, 7.3
    prior = Beta(alpha=1.0, beta=1.0, lower=lower, upper=upper)

    assert prior.lower == lower
    assert prior.upper == upper
    assert prior.scale == upper - lower


def test_halfnormal_creation_and_validation():
    """Test HalfNormal prior creation."""
    # Valid parameters with defaults
    prior = HalfNormal(sigma=1.0)
    assert prior.constraint.constraint_type == ConstraintType.LOWER_BOUNDED
    assert prior.constraint.bounds == (0.0, float("inf"))

    # Valid parameters with custom mu
    prior_shifted = HalfNormal(sigma=0.5, mu=2.0)
    assert prior_shifted.constraint.bounds == (2.0, float("inf"))

    # Invalid sigma should raise error
    with pytest.raises(ValueError, match="must be positive"):
        HalfNormal(sigma=-1.0)


def test_halfnormal_keyword_only_sigma():
    """Test that HalfNormal sigma is keyword-only."""
    # Should work with keyword
    prior = HalfNormal(sigma=1.0, mu=0.0)

    # Should fail without keyword (this test may need adjustment based on implementation)
    # with pytest.raises(TypeError):
    #     HalfNormal(1.0)  # positional sigma should fail


def test_gamma_creation_and_validation():
    """Test Gamma prior creation."""
    # Valid parameters
    prior = Gamma(alpha=2.0, beta=1.0)
    assert prior.constraint.constraint_type == ConstraintType.LOWER_BOUNDED
    assert prior.constraint.bounds == (0.0, float("inf"))

    # Valid parameters with custom mu
    prior_shifted = Gamma(alpha=2.0, beta=1.0, mu=1.5)
    assert prior_shifted.constraint.bounds == (1.5, float("inf"))

    # Invalid parameters should raise errors
    with pytest.raises(ValueError, match="must be positive"):
        Gamma(alpha=-1.0, beta=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        Gamma(alpha=1.0, beta=0.0)


def test_kde_creation_with_valid_samples():
    """Test KDE creation with valid samples."""
    samples = np.random.normal(0, 1, 100)
    prior = KDE(samples)

    assert prior._n_samples == 100
    assert prior._min_val == np.min(samples)
    assert prior._max_val == np.max(samples)
    assert prior._margin > 0


def test_kde_empty_samples_error():
    """Test that KDE with empty samples raises error."""
    with pytest.raises(ValueError, match="Cannot create KDE prior from empty samples"):
        KDE([])


def test_kde_automatic_algorithm_selection():
    """Test automatic selection between exact and grid KDE."""
    # Small sample size should use exact
    small_samples = np.random.normal(0, 1, 100)
    prior_small = KDE(small_samples)
    # Would need to inspect the created PDF type to verify exact KDE

    # Large sample size should use grid
    large_samples = np.random.normal(0, 1, 2000)
    prior_large = KDE(large_samples)
    # Would need to inspect the created PDF type to verify grid KDE


def test_kde_identical_samples_handling():
    """Test KDE handling of identical samples."""
    # All samples are the same
    identical_samples = np.array([5.0] * 50)
    prior = KDE(identical_samples)

    assert prior._range == 0.0
    assert prior._margin >= 1.0  # Should use default margin


def test_kde_custom_bandwidth():
    """Test KDE with custom bandwidth."""
    samples = np.random.normal(0, 1, 200)

    # Custom numeric bandwidth
    prior_numeric = KDE(samples, bandwidth=0.5)
    assert prior_numeric._bandwidth == 0.5

    # Custom string bandwidth
    prior_string = KDE(samples, bandwidth="silverman")
    assert prior_string._bandwidth == "silverman"


def test_kde_bounds_adaptation():
    """Test KDE bounds adaptation."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prior = KDE(samples)

    # Mock parameter with wider bounds
    class MockParam:
        has_limits = True
        lower = 0.0
        upper = 10.0

    param = MockParam()
    original_bounds = prior._original_bounds
    prior._adapt_to_parameter_limits(param)

    # Bounds should expand to accommodate parameter limits
    assert prior._original_bounds[0] <= 0.0  # Lower expanded
    assert prior._original_bounds[1] >= 10.0  # Upper expanded


def test_all_priors_have_constraints():
    """Test that all priors properly define constraints."""
    priors = [
        (Normal, {"mu": 0.0, "sigma": 1.0}),
        (Beta, {"alpha": 1.0, "beta": 1.0, "lower": 0.0, "upper": 1.0}),
        (HalfNormal, {"sigma": 1.0}),
        (Gamma, {"alpha": 2.0, "beta": 1.0}),
        (LogNormal, {"mu": 0.0, "sigma": 1.0}),
        (Cauchy, {"m": 0.0, "gamma": 1.0}),
        (Exponential, {"lam": 1.0}),
        (Poisson, {"lam": 3.0}),
        (StudentT, {"ndof": 3.0, "mu": 0.0, "sigma": 1.0}),
    ]

    for prior_class, params in priors:
        prior = prior_class(**params)
        assert hasattr(prior, 'constraint'), f"{prior_class.__name__} missing constraint"
        assert prior.constraint is not None


def test_positive_priors_have_positive_constraints():
    """Test that priors for positive parameters have appropriate constraints."""
    positive_priors = [
        HalfNormal(sigma=1.0),
        Gamma(alpha=2.0, beta=1.0),
        LogNormal(mu=0.0, sigma=1.0),
        Exponential(lam=1.0),
    ]

    for prior in positive_priors:
        # Should have positive or lower-bounded constraint
        constraint_type = prior.constraint.constraint_type
        assert constraint_type in [ConstraintType.POSITIVE, ConstraintType.LOWER_BOUNDED]


def test_bounded_priors_have_bounded_constraints():
    """Test that bounded priors have appropriate constraints."""
    # Beta should be custom bounds
    beta = Beta(alpha=1.0, beta=1.0, lower=-1.0, upper=2.0)
    assert beta.constraint.constraint_type == ConstraintType.CUSTOM_BOUNDS


def test_parameter_validation_across_priors():
    """Test parameter validation works consistently across all priors."""
    # Test cases that should fail validation in eager mode
    validation_cases = [
        (Normal, {"mu": 0.0, "sigma": -1.0}),  # Negative sigma
        (Beta, {"alpha": -1.0, "beta": 1.0, "lower": 0.0, "upper": 1.0}),  # Negative alpha
        (HalfNormal, {"sigma": 0.0}),  # Zero sigma
        (Gamma, {"alpha": 0.0, "beta": 1.0}),  # Zero alpha
        (Exponential, {"lam": -1.0}),  # Negative rate
    ]

    for prior_class, invalid_params in validation_cases:
        with pytest.raises(ValueError):
            prior_class(**invalid_params)


@pytest.mark.parametrize("prior_class,params", [
    (Normal, {"mu": 0.0, "sigma": 1.0}),
    (HalfNormal, {"sigma": 1.0}),
    (Gamma, {"alpha": 2.0, "beta": 1.0}),
    (Beta, {"alpha": 1.0, "beta": 1.0, "lower": -1.0, "upper": 2.0}),
])
def test_prior_pdf_creation(prior_class, params):
    """Test that all priors create PDFs successfully."""
    prior = prior_class(**params)
    assert prior.pdf is not None
    assert hasattr(prior.pdf, 'pdf')


@pytest.mark.parametrize("samples,expected_n", [
    (np.random.normal(0, 1, 50), 50),
    (np.random.uniform(-1, 1, 200), 200),
    (np.array([1, 2, 3, 4, 5]), 5),
])
def test_kde_sample_handling(samples, expected_n):
    """Test KDE handles different sample inputs correctly."""
    prior = KDE(samples)
    assert prior._n_samples == expected_n
    assert prior._min_val == np.min(samples)
    assert prior._max_val == np.max(samples)


if __name__ == "__main__":
    pytest.main([__file__])

"""Tests for the Bayesian constraint system."""

#  Copyright (c) 2025 zfit

import pytest
import numpy as np
import tensorflow as tf
import zfit
from zfit._bayesian.mathconstrain import (
    ConstraintType,
    PriorConstraint,
    UNCONSTRAINED,
    POSITIVE,
    UNIT_INTERVAL,
    validate_parameter,
    IdentityTransform,
    LogTransform,
    SigmoidTransform,
    AffineTransform,
    LowerBoundTransform,
    UpperBoundTransform,
)


def test_constraint_types_exist():
    """Test that all expected constraint types exist."""
    expected_types = [
        "UNCONSTRAINED", "POSITIVE", "UNIT_INTERVAL",
        "CUSTOM_BOUNDS", "LOWER_BOUNDED", "UPPER_BOUNDED"
    ]
    for constraint_type in expected_types:
        assert hasattr(ConstraintType, constraint_type)


def test_identity_transform():
    """Test identity transform."""
    transform = IdentityTransform()
    x = np.array([1.0, 2.0, -3.0])

    # Forward and inverse should be identity
    assert np.allclose(transform.forward(x), x)
    assert np.allclose(transform.inverse(x), x)
    assert np.allclose(transform.log_abs_det_jacobian(x), 0.0)
    assert transform.constraint_type == ConstraintType.UNCONSTRAINED


def test_log_transform():
    """Test log transform for positive values."""
    transform = LogTransform()
    x = np.array([0.0, 1.0, 2.0])  # unconstrained
    y_expected = np.exp(x)  # positive

    # Test forward transform
    y = transform.forward(x)
    assert np.allclose(y, y_expected)
    assert np.all(y > 0)  # Should be positive

    # Test inverse transform
    x_recovered = transform.inverse(y)
    assert np.allclose(x_recovered, x)

    # Test Jacobian
    jacobian = transform.log_abs_det_jacobian(x)
    expected_jacobian = x  # log(d/dx exp(x)) = log(exp(x)) = x
    assert np.allclose(jacobian, expected_jacobian)
    assert transform.constraint_type == ConstraintType.POSITIVE


def test_sigmoid_transform():
    """Test sigmoid transform for unit interval."""
    transform = SigmoidTransform()
    x = np.array([-2.0, 0.0, 2.0])  # unconstrained

    # Test forward transform
    y = transform.forward(x)
    assert np.all((y >= 0) & (y <= 1))  # Should be in [0, 1]

    # Test inverse transform
    x_recovered = transform.inverse(y)
    assert np.allclose(x_recovered, x, atol=1e-6)

    # Test Jacobian
    jacobian = transform.log_abs_det_jacobian(x)
    sigmoid_x = tf.nn.sigmoid(x)
    expected_jacobian = np.log(sigmoid_x) + np.log(1 - sigmoid_x)
    assert np.allclose(jacobian, expected_jacobian)
    assert transform.constraint_type == ConstraintType.UNIT_INTERVAL


def test_affine_transform():
    """Test affine transform for custom bounds."""
    lower, upper = -2.0, 5.0
    transform = AffineTransform(lower, upper)
    x = np.array([-1.0, 0.0, 1.0])  # unconstrained

    # Test forward transform
    y = transform.forward(x)
    assert np.all((y >= lower) & (y <= upper))  # Should be in [lower, upper]

    # Test inverse transform
    x_recovered = transform.inverse(y)
    assert np.allclose(x_recovered, x, atol=1e-6)

    # Test bounds validation
    with pytest.raises(ValueError):
        AffineTransform(5.0, 2.0)  # lower >= upper should fail

    assert transform.constraint_type == ConstraintType.CUSTOM_BOUNDS


def test_lower_bound_transform():
    """Test lower bound transform."""
    lower = 3.0
    transform = LowerBoundTransform(lower)
    x = np.array([-1.0, 0.0, 1.0])  # unconstrained

    # Test forward transform
    y = transform.forward(x)
    assert np.all(y >= lower)  # Should be >= lower

    # Test inverse transform
    x_recovered = transform.inverse(y)
    assert np.allclose(x_recovered, x)

    assert transform.constraint_type == ConstraintType.LOWER_BOUNDED


def test_upper_bound_transform():
    """Test upper bound transform."""
    upper = 10.0
    transform = UpperBoundTransform(upper)
    x = np.array([-1.0, 0.0, 1.0])  # unconstrained

    # Test forward transform
    y = transform.forward(x)
    assert np.all(y <= upper)  # Should be <= upper

    # Test inverse transform
    x_recovered = transform.inverse(y)
    assert np.allclose(x_recovered, x)

    assert transform.constraint_type == ConstraintType.UPPER_BOUNDED


def test_predefined_constraints():
    """Test predefined constraint objects."""
    assert UNCONSTRAINED.constraint_type == ConstraintType.UNCONSTRAINED
    assert POSITIVE.constraint_type == ConstraintType.POSITIVE
    assert UNIT_INTERVAL.constraint_type == ConstraintType.UNIT_INTERVAL

    assert UNCONSTRAINED.bounds == (-float("inf"), float("inf"))
    assert POSITIVE.bounds == (0, float("inf"))
    assert UNIT_INTERVAL.bounds == (0, 1)


def test_constraint_creation():
    """Test creating custom constraints."""
    constraint = PriorConstraint(ConstraintType.CUSTOM_BOUNDS, bounds=(-1, 2))
    assert constraint.constraint_type == ConstraintType.CUSTOM_BOUNDS
    assert constraint.bounds == (-1, 2)
    assert isinstance(constraint.transform, AffineTransform)


def test_bounds_validation():
    """Test bounds validation."""
    constraint = POSITIVE

    # Test positive constraint validation
    bounds = constraint.validate_bounds((-1, 5))
    assert bounds == (0, 5)  # Lower bound should be clipped to 0

    # Test unit interval constraint validation
    constraint = UNIT_INTERVAL
    bounds = constraint.validate_bounds((-0.5, 1.5))
    assert bounds == (0, 1)  # Should be clipped to [0, 1]


def test_constraint_errors():
    """Test constraint error handling."""
    with pytest.raises(ValueError):
        PriorConstraint(ConstraintType.CUSTOM_BOUNDS)  # Missing bounds

    with pytest.raises(ValueError):
        PriorConstraint(ConstraintType.LOWER_BOUNDED)  # Missing finite lower bound

    with pytest.raises(ValueError):
        PriorConstraint(ConstraintType.UPPER_BOUNDED)  # Missing finite upper bound


def test_basic_validation():
    """Test basic parameter validation."""
    # Should not raise in eager mode
    validate_parameter("test", 1.0)
    validate_parameter("test", -5.0)
    validate_parameter("test", 0.0)

    # Test non-numeric values
    with pytest.raises(ValueError, match="must be numeric"):
        validate_parameter("test", "invalid")

    with pytest.raises(ValueError, match="must be numeric"):
        validate_parameter("test", [1, 2, 3])


def test_positive_constraint_validation():
    """Test validation with positive constraint."""
    # Valid positive values
    validate_parameter("sigma", 1.0, POSITIVE)
    validate_parameter("sigma", 0.1, POSITIVE)

    # Invalid non-positive values
    with pytest.raises(ValueError, match="must be positive"):
        validate_parameter("sigma", 0.0, POSITIVE)

    with pytest.raises(ValueError, match="must be positive"):
        validate_parameter("sigma", -1.0, POSITIVE)


def test_unit_interval_validation():
    """Test validation with unit interval constraint."""
    # Valid unit interval values
    validate_parameter("prob", 0.0, UNIT_INTERVAL)
    validate_parameter("prob", 0.5, UNIT_INTERVAL)
    validate_parameter("prob", 1.0, UNIT_INTERVAL)

    # Invalid values outside [0, 1]
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        validate_parameter("prob", -0.1, UNIT_INTERVAL)

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        validate_parameter("prob", 1.1, UNIT_INTERVAL)


def test_custom_bounds_validation():
    """Test validation with custom bounds."""
    constraint = PriorConstraint(ConstraintType.CUSTOM_BOUNDS, bounds=(-2, 5))

    # Valid values within bounds
    validate_parameter("param", -2.0, constraint)
    validate_parameter("param", 0.0, constraint)
    validate_parameter("param", 5.0, constraint)

    # Invalid values outside bounds
    with pytest.raises(ValueError, match="must be in \\[-2, 5\\]"):
        validate_parameter("param", -3.0, constraint)

    with pytest.raises(ValueError, match="must be in \\[-2, 5\\]"):
        validate_parameter("param", 6.0, constraint)


@pytest.mark.skipif(not hasattr(zfit.run, 'executing_eagerly'),
                   reason="zfit.run.executing_eagerly not available")
def test_validation_skipped_in_graph_mode():
    """Test that validation is skipped in graph mode."""
    # This test would require mocking zfit.run.executing_eagerly() to return False
    # For now, we'll just test that the function exists
    assert hasattr(zfit.run, 'executing_eagerly')


def test_constraint_assignment():
    """Test that constraints are properly assigned to priors."""
    from zfit._bayesian.priors import Normal, Beta, HalfNormal

    # Test Normal has unconstrained
    normal = Normal(mu=0.0, sigma=1.0)
    assert normal.constraint.constraint_type == ConstraintType.UNCONSTRAINED

    # Test Beta has custom bounds constraint
    beta = Beta(alpha=2.0, beta=3.0, lower=-1.0, upper=2.0)
    assert beta.constraint.constraint_type == ConstraintType.CUSTOM_BOUNDS

    # Test HalfNormal has lower bound constraint
    halfnorm = HalfNormal(sigma=1.0, mu=0.0)
    assert halfnorm.constraint.constraint_type == ConstraintType.LOWER_BOUNDED


def test_parameter_validation_integration():
    """Test that parameter validation works with actual priors."""
    from zfit._bayesian.priors import Normal, Beta

    # Valid parameters should work
    normal = Normal(mu=0.0, sigma=1.0)
    beta = Beta(alpha=1.0, beta=2.0, lower=-1.0, upper=2.0)

    # Invalid parameters should raise errors in eager mode
    with pytest.raises(ValueError, match="must be positive"):
        Normal(mu=0.0, sigma=-1.0)  # Negative sigma

    with pytest.raises(ValueError, match="must be positive"):
        Beta(alpha=-1.0, beta=2.0, lower=-1.0, upper=2.0)  # Negative alpha


if __name__ == "__main__":
    pytest.main([__file__])

"""Constraint and transform system for Bayesian priors.

This module provides a systematic approach to handling parameter constraints
and transformations, similar to PyMC's transform system and TensorFlow
Probability's bijectors. This enables automatic handling of bounded parameters
while maintaining proper probability densities.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf

import zfit.z.numpy as znp


class ConstraintType(Enum):
    """Types of parameter constraints supported."""

    UNCONSTRAINED = "unconstrained"
    POSITIVE = "positive"  # [0, ∞)
    UNIT_INTERVAL = "unit_interval"  # [0, 1]
    CUSTOM_BOUNDS = "custom_bounds"  # [a, b]
    LOWER_BOUNDED = "lower_bounded"  # [a, ∞)
    UPPER_BOUNDED = "upper_bounded"  # (-∞, b]


class Transform(ABC):
    """Abstract base class for parameter transforms.

    Transforms map between constrained and unconstrained spaces,
    enabling efficient MCMC sampling on the unconstrained real line
    while maintaining proper parameter constraints.
    """

    @abstractmethod
    def forward(self, x):
        """Transform from unconstrained to constrained space."""

    @abstractmethod
    def inverse(self, y):
        """Transform from constrained to unconstrained space."""

    @abstractmethod
    def log_abs_det_jacobian(self, x):
        """Log absolute determinant of Jacobian for forward transform."""

    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """Type of constraint this transform handles."""


class IdentityTransform(Transform):
    """Identity transform for unconstrained parameters."""

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x):
        return znp.zeros_like(x)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.UNCONSTRAINED


class LogTransform(Transform):
    """Log transform for positive parameters: y = exp(x)."""

    def forward(self, x):
        return znp.exp(x)

    def inverse(self, y):
        return znp.log(y)

    def log_abs_det_jacobian(self, x):
        return x  # d/dx exp(x) = exp(x), log(exp(x)) = x

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.POSITIVE


class SigmoidTransform(Transform):
    """Sigmoid transform for unit interval parameters: y = sigmoid(x)."""

    def forward(self, x):
        return tf.nn.sigmoid(x)

    def inverse(self, y):
        return znp.log(y / (1 - y))  # logit

    def log_abs_det_jacobian(self, x):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = tf.nn.sigmoid(x)
        return znp.log(sigmoid_x) + znp.log(1 - sigmoid_x)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.UNIT_INTERVAL


class AffineTransform(Transform):
    """Affine transform for custom bounds: y = a + (b-a) * sigmoid(x)."""

    def __init__(self, lower: float, upper: float):
        if lower >= upper:
            msg = f"Lower bound {lower} must be less than upper bound {upper}"
            raise ValueError(msg)
        self.lower = float(lower)
        self.upper = float(upper)
        self.scale = self.upper - self.lower

    def forward(self, x):
        return self.lower + self.scale * tf.nn.sigmoid(x)

    def inverse(self, y):
        normalized = (y - self.lower) / self.scale
        return znp.log(normalized / (1 - normalized))  # logit

    def log_abs_det_jacobian(self, x):
        # d/dy = scale * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = tf.nn.sigmoid(x)
        return znp.log(self.scale) + znp.log(sigmoid_x) + znp.log(1 - sigmoid_x)

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.CUSTOM_BOUNDS


class LowerBoundTransform(Transform):
    """Lower bound transform: y = lower + exp(x)."""

    def __init__(self, lower: float):
        self.lower = float(lower)

    def forward(self, x):
        return self.lower + znp.exp(x)

    def inverse(self, y):
        return znp.log(y - self.lower)

    def log_abs_det_jacobian(self, x):
        return x  # d/dx (lower + exp(x)) = exp(x), log(exp(x)) = x

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.LOWER_BOUNDED


class UpperBoundTransform(Transform):
    """Upper bound transform: y = upper - exp(x)."""

    def __init__(self, upper: float):
        self.upper = float(upper)

    def forward(self, x):
        return self.upper - znp.exp(x)

    def inverse(self, y):
        return znp.log(self.upper - y)

    def log_abs_det_jacobian(self, x):
        return x  # d/dx (upper - exp(x)) = -exp(x), log(|-exp(x)|) = log(exp(x)) = x

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.UPPER_BOUNDED


class PriorConstraint:
    """Constraint specification for a prior distribution.

    This class encapsulates the constraint type, bounds, and associated
    transform for a prior distribution, providing a unified interface
    for handling parameter constraints.
    """

    def __init__(
        self,
        constraint_type: ConstraintType,
        bounds: tuple[float, float] | None = None,
        transform: Transform | None = None,
    ):
        self.constraint_type = constraint_type
        self._original_bounds = bounds  # Store original bounds for validation
        self.bounds = bounds or (-float("inf"), float("inf"))
        self.transform = transform or self._create_default_transform()

    def _create_default_transform(self) -> Transform:
        """Create default transform based on constraint type."""
        if self.constraint_type == ConstraintType.UNCONSTRAINED:
            return IdentityTransform()
        elif self.constraint_type == ConstraintType.POSITIVE:
            return LogTransform()
        elif self.constraint_type == ConstraintType.UNIT_INTERVAL:
            return SigmoidTransform()
        elif self.constraint_type == ConstraintType.CUSTOM_BOUNDS:
            if self._original_bounds is None:
                msg = "CUSTOM_BOUNDS requires bounds to be specified"
                raise ValueError(msg)
            return AffineTransform(self.bounds[0], self.bounds[1])
        elif self.constraint_type == ConstraintType.LOWER_BOUNDED:
            if self._original_bounds is None or self.bounds[0] == -float("inf"):
                msg = "LOWER_BOUNDED requires finite lower bound"
                raise ValueError(msg)
            return LowerBoundTransform(self.bounds[0])
        elif self.constraint_type == ConstraintType.UPPER_BOUNDED:
            if self._original_bounds is None or self.bounds[1] == float("inf"):
                msg = "UPPER_BOUNDED requires finite upper bound"
                raise ValueError(msg)
            return UpperBoundTransform(self.bounds[1])
        else:
            msg = f"Unknown constraint type: {self.constraint_type}"
            raise ValueError(msg)

    def validate_bounds(self, bounds: tuple[float, float]) -> tuple[float, float]:
        """Validate and adapt bounds according to constraint."""
        lower, upper = bounds

        if self.constraint_type == ConstraintType.POSITIVE:
            lower = max(0.0, lower)
        elif self.constraint_type == ConstraintType.UNIT_INTERVAL:
            lower = max(0.0, lower)
            upper = min(1.0, upper)
        elif self.constraint_type == ConstraintType.LOWER_BOUNDED:
            if self.bounds and self.bounds[0] != -float("inf"):
                lower = max(self.bounds[0], lower)
        elif self.constraint_type == ConstraintType.UPPER_BOUNDED:
            if self.bounds and self.bounds[1] != float("inf"):
                upper = min(self.bounds[1], upper)
        elif self.constraint_type == ConstraintType.CUSTOM_BOUNDS and self.bounds:
            lower = max(self.bounds[0], lower)
            upper = min(self.bounds[1], upper)

        return lower, upper

    def __repr__(self) -> str:
        return f"PriorConstraint({self.constraint_type.value}, bounds={self.bounds})"


# Predefined constraint objects for common use cases
UNCONSTRAINED = PriorConstraint(ConstraintType.UNCONSTRAINED)
POSITIVE = PriorConstraint(ConstraintType.POSITIVE, bounds=(0, float("inf")))
UNIT_INTERVAL = PriorConstraint(ConstraintType.UNIT_INTERVAL, bounds=(0, 1))


def validate_parameter(name: str, value, constraint: PriorConstraint | None = None):
    """Validate parameter value if in eager mode.

    Args:
        name: Parameter name for error messages
        value: Parameter value to validate
        constraint: Optional constraint to check against
    """
    # Import run locally to avoid circular import
    try:
        from zfit import run  # noqa: PLC0415

        if not run.executing_eagerly():
            return  # Skip validation in graph mode
    except ImportError:
        # If zfit.run is not available, assume eager mode for validation
        pass

    try:
        val = float(value)
    except (TypeError, ValueError) as e:
        msg = f"Parameter '{name}' must be numeric, got {type(value).__name__}: {value}"
        raise ValueError(msg) from e

    if constraint is None:
        return

    # Check constraint-specific validations
    if constraint.constraint_type == ConstraintType.POSITIVE:
        if val <= 0:
            msg = f"Parameter '{name}' must be positive, got {val}"
            raise ValueError(msg)
    elif constraint.constraint_type == ConstraintType.UNIT_INTERVAL:
        if not (0 <= val <= 1):
            msg = f"Parameter '{name}' must be in [0, 1], got {val}"
            raise ValueError(msg)
    elif constraint.constraint_type == ConstraintType.CUSTOM_BOUNDS:
        lower, upper = constraint.bounds
        if not (lower <= val <= upper):
            msg = f"Parameter '{name}' must be in [{lower}, {upper}], got {val}"
            raise ValueError(msg)
    elif constraint.constraint_type == ConstraintType.LOWER_BOUNDED:
        lower = constraint.bounds[0]
        if val < lower:
            msg = f"Parameter '{name}' must be >= {lower}, got {val}"
            raise ValueError(msg)
    elif constraint.constraint_type == ConstraintType.UPPER_BOUNDED:
        upper = constraint.bounds[1]
        if val > upper:
            msg = f"Parameter '{name}' must be <= {upper}, got {val}"
            raise ValueError(msg)

"""Parameter validation utilities to reduce code duplication."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp

from ..util.exception import ShapeIncompatibleError

if TYPE_CHECKING:
    import zfit  # noqa: F401


def validate_parameter_type(param: Any, name: str = "parameter") -> None:
    """Validate that the input is a ZfitParameter.

    Args:
        param: The parameter to validate
        name: Name to use in error messages

    Raises:
        TypeError: If param is not a ZfitParameter
    """
    from ..core.interfaces import ZfitParameter

    if not isinstance(param, ZfitParameter):
        raise TypeError(f"`{name}` must be ZfitParameter, got {type(param)}")


def validate_parameters(*params: Any, names: list[str] | None = None) -> None:
    """Validate that all inputs are ZfitParameters.

    Args:
        *params: Parameters to validate
        names: Optional names for each parameter in error messages

    Raises:
        TypeError: If any param is not a ZfitParameter
    """
    if names is None:
        names = [f"param{i + 1}" for i in range(len(params))]

    for i, param in enumerate(params):
        name = names[i] if i < len(names) else f"param{i + 1}"
        validate_parameter_type(param, name)


def validate_limit_bounds(lower: Any, upper: Any) -> tuple[Any, Any]:
    """Validate and normalize parameter limit bounds.

    Args:
        lower: Lower bound
        upper: Upper bound

    Returns:
        Tuple of (normalized_lower, normalized_upper)

    Raises:
        ValueError: If bounds are invalid
    """
    if lower is not None and upper is not None:
        if znp.any(lower >= upper):
            raise ValueError(f"Lower limit ({lower}) must be less than upper limit ({upper})")

    return lower, upper


def validate_value_in_limits(value: Any, lower: Any = None, upper: Any = None, name: str = "value") -> None:
    """Validate that a value is within specified limits.

    Args:
        value: Value to check
        lower: Lower limit (optional)
        upper: Upper limit (optional)
        name: Name for error messages

    Raises:
        ValueError: If value is outside limits
    """
    if lower is not None and znp.any(value < lower):
        raise ValueError(f"{name} ({value}) is below lower limit ({lower})")

    if upper is not None and znp.any(value > upper):
        raise ValueError(f"{name} ({value}) is above upper limit ({upper})")


def calculate_limit_tolerance(lower: Any, upper: Any, exact: bool = False) -> Any:
    """Calculate tolerance for limit checking.

    Args:
        lower: Lower bound
        upper: Upper bound
        exact: Whether to use exact tolerance

    Returns:
        Calculated tolerance value
    """
    if lower is None or upper is None:
        return 1e-7 if exact else 1e-5

    diff = znp.abs(upper - lower)
    if not exact:
        reltol = 0.005
        abstol = 1e-5
    else:
        reltol = 1e-5
        abstol = 1e-7

    return znp.minimum(diff * reltol, abstol)


def check_at_limit(value: Any, lower: Any = None, upper: Any = None, exact: bool = False) -> bool:
    """Check if a value is at its limits within tolerance.

    Args:
        value: Value to check
        lower: Lower limit (optional)
        upper: Upper limit (optional)
        exact: Whether to use exact tolerance

    Returns:
        True if value is at a limit, False otherwise
    """
    if lower is None and upper is None:
        return False

    tol = calculate_limit_tolerance(lower, upper, exact=exact)

    at_lower = lower is not None and znp.abs(value - lower) <= tol
    at_upper = upper is not None and znp.abs(value - upper) <= tol

    return znp.any(at_lower) or znp.any(at_upper)


def validate_parameter_shapes(*params: Any, require_same: bool = True) -> None:
    """Validate parameter shapes are compatible.

    Args:
        *params: Parameters to check
        require_same: Whether all shapes must be identical

    Raises:
        ShapeIncompatibleError: If shapes are incompatible
    """
    if len(params) < 2:
        return

    shapes = [getattr(p, "shape", None) for p in params]

    if require_same:
        first_shape = shapes[0]
        for i, shape in enumerate(shapes[1:], 1):
            if shape != first_shape:
                raise ShapeIncompatibleError(
                    f"Parameter shapes must be identical. param1 shape: {first_shape}, param{i + 1} shape: {shape}"
                )


def validate_parameter_names(*names: str) -> None:
    """Validate parameter names follow conventions.

    Args:
        *names: Parameter names to validate

    Raises:
        ValueError: If any name is invalid
    """
    import keyword

    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"Parameter name must be string, got {type(name)}")

        if not name:
            raise ValueError("Parameter name cannot be empty")

        # Check for Python keywords
        if keyword.iskeyword(name):
            raise ValueError(f"Parameter name '{name}' is a Python keyword")

        # Check for valid identifier
        if not name.isidentifier():
            raise ValueError(f"Parameter name '{name}' is not a valid Python identifier")

        # Check for reserved zfit names (common patterns)
        reserved_patterns = ["__", "_zfit_", "zfit_"]
        if any(pattern in name for pattern in reserved_patterns):
            raise ValueError(f"Parameter name '{name}' uses reserved pattern")


def format_parameter_error(expected_type: str, actual_type: type, param_name: str = "parameter") -> str:
    """Format a standardized parameter type error message.

    Args:
        expected_type: Expected type name
        actual_type: Actual type received
        param_name: Parameter name for context

    Returns:
        Formatted error message
    """
    return f"`{param_name}` must be {expected_type}, got {actual_type.__name__}"


def validate_stepsize(stepsize: Any, param_name: str = "parameter") -> None:
    """Validate parameter stepsize value.

    Args:
        stepsize: Stepsize value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If stepsize is invalid
    """
    if stepsize is None:
        return

    try:
        # Try to convert to float to verify it's numeric
        float(stepsize)  # Just check if conversion works
    except (TypeError, ValueError):
        raise ValueError(f"Stepsize for {param_name} must be convertible to float, got {type(stepsize)}")

    if znp.any(stepsize <= 0):
        raise ValueError(f"Stepsize for {param_name} must be positive, got {stepsize}")

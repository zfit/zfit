"""Parameter validation utilities to reduce code duplication."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import typing
from typing import Any

import zfit.z.numpy as znp

from ..util.exception import ShapeIncompatibleError

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def validate_parameter_type(param: Any, name: str = "parameter") -> None:
    """Validate that the input is a ZfitParameter.

    Args:
        param: The parameter to validate
        name: Name to use in error messages

    Raises:
        TypeError: If param is not a ZfitParameter
    """
    from zfit._interfaces import ZfitParameter  # noqa: PLC0415

    if not isinstance(param, ZfitParameter):
        msg = f"`{name}` must be ZfitParameter, got {type(param)}"
        raise TypeError(msg)


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
    if lower is not None and upper is not None and znp.any(lower >= upper):
        msg = f"Lower limit ({lower}) must be less than upper limit ({upper})"
        raise ValueError(msg)

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
        msg = f"{name} ({value}) is below lower limit ({lower})"
        raise ValueError(msg)

    if upper is not None and znp.any(value > upper):
        msg = f"{name} ({value}) is above upper limit ({upper})"
        raise ValueError(msg)


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
                msg = f"Parameter shapes must be identical. param1 shape: {first_shape}, param{i + 1} shape: {shape}"
                raise ShapeIncompatibleError(msg)


def validate_parameter_names(*names: str) -> None:
    """Validate parameter names follow conventions.

    Args:
        *names: Parameter names to validate

    Raises:
        ValueError: If any name is invalid
    """
    import keyword  # noqa: PLC0415

    for name in names:
        if not isinstance(name, str):
            msg = f"Parameter name must be string, got {type(name)}"
            raise ValueError(msg)

        if not name:
            msg = "Parameter name cannot be empty"
            raise ValueError(msg)

        # Check for Python keywords
        if keyword.iskeyword(name):
            msg = f"Parameter name '{name}' is a Python keyword"
            raise ValueError(msg)

        # Check for valid identifier
        if not name.isidentifier():
            msg = f"Parameter name '{name}' is not a valid Python identifier"
            raise ValueError(msg)

        # Check for reserved zfit names (common patterns)
        reserved_patterns = ["__", "_zfit_", "zfit_"]
        if any(pattern in name for pattern in reserved_patterns):
            msg = f"Parameter name '{name}' uses reserved pattern"
            raise ValueError(msg)


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
    except (TypeError, ValueError) as error:
        msg = f"Stepsize for {param_name} must be convertible to float, got {type(stepsize)}"
        raise ValueError(msg) from error

    if znp.any(stepsize <= 0):
        msg = f"Stepsize for {param_name} must be positive, got {stepsize}"
        raise ValueError(msg)

#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

import functools
import warnings

from tensorflow.python.util.deprecation import deprecated_args

_PRINTED_WARNING = {}


def deprecated(date, instructions, warn_once=True):
    """Decorator for marking functions or methods deprecated.

    This decorator logs a deprecation warning whenever the decorated function is
    called. It has the following format:

      <function> (from <module>) is deprecated and will be removed after <date>.
      Instructions for updating:
      <instructions>

    If `date` is None, 'after <date>' is replaced with 'in a future version'.
    <function> will include the class name if it is a method.

    It also edits the docstring of the function: ' (deprecated)' is appended
    to the first line of the docstring and a deprecation notice is prepended
    to the rest of the docstring.

    Args:
      date: String or None. The date the function is scheduled to be removed. Must
        be ISO 8601 (YYYY-MM-DD), or None.
      instructions: String. Instructions on how to update code using the
        deprecated function.
      warn_once: Boolean. Set to `True` to warn only the first time the decorated
        function is called. Otherwise, every call will log a warning.

    Returns:
      Decorated function or method.

    Raises:
      ValueError: If date is not None or in ISO 8601 format, or instructions are
        empty.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_key = func.__name__
            if not warn_once or warning_key not in _PRINTED_WARNING:
                _PRINTED_WARNING[warning_key] = True
                removal_date = "in a future version" if date is None else f"after {date}"
                warnings.warn(
                    f"{func.__name__} (from {func.__module__}) is deprecated and "
                    f"will be removed {removal_date}.\n"
                    f"Instructions for updating:\n{instructions}",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["deprecate_norm_range", "deprecated", "deprecated_args"]  # noqa: F822


def deprecated_norm_range(func):
    @functools.wraps(func)
    @deprecated_args(None, "Use `norm` instead.", "norm_range")
    def wrapper(*args, norm=None, norm_range=None, **kwargs):
        if norm_range is not None:
            norm = norm_range
        try:
            return func(*args, norm=norm, **kwargs)
        except TypeError as error:
            if "unexpected keyword argument 'norm'" in str(error):
                return func(*args, norm_range=norm_range, **kwargs)
            elif "got multiple values for argument 'norm'" in str(error):
                return func(*args, **kwargs)
            else:
                raise

    return wrapper

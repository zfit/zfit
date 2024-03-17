#  Copyright (c) 2022 zfit
import functools

from tensorflow.python.util.deprecation import deprecated, deprecated_args

__all__ = "deprecated", "deprecated_args", "deprecate_norm_range"  # noqa


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

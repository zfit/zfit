#  Copyright (c) 2025 zfit
from __future__ import annotations

import functools
import typing
from typing import Any

import tensorflow as tf

import zfit.z.numpy as _znp

from ..settings import ztypes
from ..util.exception import BreakingAPIChangeError
from .tools import _auto_upcast

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def exp(x) -> tf.Tensor:
    return _auto_upcast(_znp.exp(x=x))


@functools.wraps(tf.convert_to_tensor)
def convert_to_tensor(value, dtype=ztypes.float) -> tf.Tensor:
    return _znp.asarray(value, dtype)


def random_normal(*_, **__) -> typing.NoReturn:
    msg = "Use z.random.normal instead."
    raise BreakingAPIChangeError(msg)


def random_uniform(*_, **__) -> typing.NoReturn:
    msg = "Use z.random.uniform instead."
    raise BreakingAPIChangeError(msg)


def random_poisson(*_, **__) -> typing.NoReturn:
    msg = "Use z.random.poisson instead."
    raise BreakingAPIChangeError(msg)


def square(x, name=None) -> tf.Tensor:
    return _auto_upcast(tf.square(x, name))


def sqrt(x, name=None) -> tf.Tensor:
    return _auto_upcast(tf.sqrt(x, name=name))


def pow(x, y, name=None) -> tf.Tensor:
    return _auto_upcast(tf.pow(x, y, name=name))


def complex(real, imag, name=None) -> tf.Tensor:
    real = _auto_upcast(real)
    imag = _auto_upcast(imag)
    return _auto_upcast(tf.complex(real=real, imag=imag, name=name))


def check_numerics(tensor: Any, message: Any, name: Any = None) -> tf.Operation:
    """Check whether a tensor is finite and not NaN. Extends TF by accepting complex types as well.

    Args:
        tensor:
        message:
        name:

    Returns:
        A TensorFlow operation that checks if the tensor is valid
    """
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    if tf.as_dtype(tensor.dtype).is_complex:
        real_check = tf.debugging.check_numerics(tensor=_znp.real(tensor), message=message, name=name)
        imag_check = tf.debugging.check_numerics(tensor=_znp.imag(tensor), message=message, name=name)
        check_op = tf.group(real_check, imag_check)
    else:
        check_op = tf.debugging.check_numerics(tensor=tensor, message=message, name=name)
    return check_op


def assert_all_finite(t: tf.Tensor, msg: str | None = None) -> tf.Operation:
    """Assert that all elements of a tensor are finite."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_all_finite(t, msg)


def assert_positive(t: tf.Tensor, msg: str | None = None) -> tf.Operation:
    """Assert that all elements of a tensor are positive."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_positive(t, msg)


def assert_non_negative(t: tf.Tensor, msg: str | None = None) -> tf.Operation:
    """Assert that all elements of a tensor are non-negative."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_non_negative(t, msg)


def assert_equal(t1: tf.Tensor, t2: tf.Tensor, message: str | None = None) -> tf.Operation:
    """Assert that two tensors are equal."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_equal(t1, t2, message)


def assert_greater_equal(x: tf.Tensor, y: tf.Tensor, msg: str | None = None) -> tf.Operation | None:
    """Assert that two tensors are equal or greater."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_greater_equal(x, y, msg)


def assert_greater(x: tf.Tensor, y: tf.Tensor, message: str | None = None) -> tf.Operation | None:
    """Assert that two tensors are greater."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_greater(x, y, message)


def assert_less(x: tf.Tensor, y: tf.Tensor, message: str | None = None) -> tf.Operation | None:
    """Assert that two tensors are less."""
    from .. import run  # noqa: PLC0415

    if not run.numeric_checks:
        return None
    return tf.debugging.assert_less(x, y, message)


reduce_sum = _znp.sum

reduce_prod = _znp.prod

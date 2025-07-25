#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

SWITCH_ON = True


def is_tensor(x) -> bool:
    return tf.is_tensor(x)


def has_tensor(x) -> bool:
    return any(tf.is_tensor(t) for t in tf.nest.flatten(x))


def allclose_anyaware(x, y, rtol=1e-5, atol=1e-8) -> bool:
    """Tests if x and y are close by first testing equality (with numpy), then within the limits.

    The prepended equality test allow for ANY objects to compare positively if the x and y have the shape (1, n)
    with n arbitrary

    Args:
        x:
        y:
        rtol:
        atol:

    Returns:
        bool: True if x and y are close, False otherwise
    """
    if not SWITCH_ON or has_tensor([x, y]):
        return znp.all(znp.less_equal(znp.abs(x - y), znp.abs(y) * rtol + atol))

    x = np.array(x)
    y = np.array(y)
    if any(ar.dtype == object for ar in (x, y)):
        from zfit.core.space import LimitRangeDefinition  # noqa: PLC0415

        equal = []
        for x1, y1 in zip(x[0], y[0], strict=True):
            if isinstance(x1, LimitRangeDefinition) or isinstance(y1, LimitRangeDefinition):
                equal.append(x1 < y1 or x1 > y1)
            else:
                equal.append(np.allclose(x1, y1, rtol=rtol, atol=atol))
        allclose = np.array(equal)[None, :]
    else:
        allclose = np.allclose(x, y, rtol=rtol, atol=atol)

    return allclose


def broadcast_to(input, shape) -> tf.Tensor:
    if not SWITCH_ON or is_tensor(input):
        return tf.broadcast_to(input, shape)
    return np.broadcast_to(input, shape)


def expand_dims(input, axis) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(input):
        return znp.expand_dims(input, axis)
    return np.expand_dims(input, axis)


def reduce_prod(input_tensor, axis=None, keepdims=None) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(input_tensor):
        return znp.prod(input_tensor, axis, keepdims=keepdims)
    elif keepdims is None:
        return np.prod(input_tensor, axis)
    return np.prod(input_tensor, axis, keepdims=keepdims)


def equal(x, y) -> tf.Tensor:
    if not SWITCH_ON or is_tensor(x) or is_tensor(y):
        return znp.equal(x, y)
    return np.equal(x, y)


def reduce_all(input_tensor, axis=None) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(input_tensor):
        if axis is None:
            input_tensor = [znp.reshape(ar, (-1,)) for ar in tf.nest.flatten(input_tensor)]
        return znp.all(input_tensor, axis)
    out = np.all(input_tensor, axis)
    if out.shape == (1,):
        out = out[0]
    return out


def reduce_any(input_tensor, axis=None) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(input_tensor):
        if axis is None:
            input_tensor = [znp.reshape(ar, (-1,)) for ar in tf.nest.flatten(input_tensor)]
        return znp.any(input_tensor, axis)

    out = np.any(input_tensor, axis)
    if out.shape == (1,):
        out = out[0]
    return out


def logical_and(x, y) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(x) or has_tensor(y):
        return znp.logical_and(x, y)
    return np.logical_and(x, y)


def logical_or(x, y) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(x) or has_tensor(y):
        return znp.logical_or(x, y)
    return np.logical_or(x, y)


def less_equal(x, y) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(x) or has_tensor(y):
        return znp.less_equal(x, y)
    return np.less_equal(x, y)


def greater_equal(x, y) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(x) or has_tensor(y):
        return znp.greater_equal(x, y)
    return np.greater_equal(x, y)


def gather(x, indices=None, axis=None) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(x):
        return tf.gather(x, indices=indices, axis=axis)
    return np.take(x, indices=indices, axis=axis)


def concat(values, axis) -> tf.Tensor:
    if not SWITCH_ON or has_tensor(values):
        return znp.concatenate(values, axis=axis)
    return np.concatenate(values, axis=axis)


def _try_convert_numpy(tensorlike) -> np.ndarray:
    if hasattr(tensorlike, "numpy"):
        tensorlike = tensorlike.numpy()

    if not isinstance(tensorlike, np.ndarray):
        from zfit.util.exception import CannotConvertToNumpyError  # noqa: PLC0415

        msg = (
            f"Cannot convert {tensorlike} to a Numpy array. This may be because the"
            f" object is a Tensor and the function is called in Graph mode (e.g. in"
            f"a `z.function` decorated function.\n"
            f"If this error appears and is not understandable, it is most likely a bug."
            f" Please open an issue on Github."
        )
        raise CannotConvertToNumpyError(msg)
    return tensorlike

#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

import tensorflow as tf

from zfit.settings import upcast_ztypes

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def _auto_upcast(tensor: tf.Tensor) -> tf.Tensor:
    import zfit.z.numpy as znp  # noqa: PLC0415

    if isinstance(tensor, tf.Tensor):
        new_dtype = upcast_ztypes[tensor.dtype]
        if new_dtype != tensor.dtype:
            tensor = znp.asarray(tensor, dtype=new_dtype)
    return tensor


def _get_ndims(x) -> int:
    """Returns the number of dimensions of a TensorFlow Tensor or a numpy array like object.

    Both objects have a different method to get the number of dimensions. Numpy arrays have a .ndim attribute,
    whereas TensorFlow tensors have a .shape.ndims and a .rank attribute.

    Args:
        x: A TensorFlow Tensor or a numpy array like object to extract the number of dimensions from.

    Returns:
        The number of dimensions of the input object.
    """
    try:
        return x.ndim  # numpy like
    except AttributeError:
        return x.shape.ndims  # tensorflow like

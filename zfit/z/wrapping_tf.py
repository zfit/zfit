#  Copyright (c) 2023 zfit

import functools
from typing import Any

import tensorflow as tf

import zfit.z.numpy as znp
from .tools import _auto_upcast
from ..settings import ztypes
from ..util.deprecation import deprecated


def exp(x):
    return _auto_upcast(znp.exp(x=x))


@functools.wraps(tf.convert_to_tensor)
def convert_to_tensor(
    value, dtype=ztypes.float, name=None, preferred_dtype=ztypes.float
):
    value = tf.cast(value, dtype=dtype)
    return tf.convert_to_tensor(
        value=value, dtype=dtype, name=name, dtype_hint=preferred_dtype
    )


@deprecated(None, "Use z.random.normal instead.")
def random_normal(
    shape, mean=0.0, stddev=1.0, dtype=ztypes.float, seed=None, name=None
):
    return z.random.get_prng().normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name
    )


@deprecated(None, "Use z.random.uniform instead.")
def random_uniform(
    shape, minval=0, maxval=None, dtype=ztypes.float, seed=None, name=None
):
    return z.random.get_prng().uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed, name=name
    )


@deprecated(None, "Use z.random.poisson instead.")
def random_poisson(
    lam: Any,
    shape: Any,
    dtype: tf.DType = ztypes.float,
    seed: Any = None,
    name: Any = None,
):
    return tf.random.poisson(lam=lam, shape=shape, dtype=dtype, seed=seed, name=name)


def square(x, name=None):
    return _auto_upcast(tf.square(x, name))


def sqrt(x, name=None):
    return _auto_upcast(tf.sqrt(x, name=name))


def pow(x, y, name=None):
    return _auto_upcast(tf.pow(x, y, name=name))


def complex(real, imag, name=None):
    real = _auto_upcast(real)
    imag = _auto_upcast(imag)
    return _auto_upcast(tf.complex(real=real, imag=imag, name=name))


def check_numerics(tensor: Any, message: Any, name: Any = None):
    """Check whether a tensor is finite and not NaN. Extends TF by accepting complex types as well.

    Args:
        tensor:
        message:
        name:

    Returns:
    """
    if tf.as_dtype(tensor.dtype).is_complex:
        real_check = tf.debugging.check_numerics(
            tensor=znp.real(tensor), message=message, name=name
        )
        imag_check = tf.debugging.check_numerics(
            tensor=znp.imag(tensor), message=message, name=name
        )
        check_op = tf.group(real_check, imag_check)
    else:
        check_op = tf.debugging.check_numerics(
            tensor=tensor, message=message, name=name
        )
    return check_op


reduce_sum = znp.sum

reduce_prod = znp.prod

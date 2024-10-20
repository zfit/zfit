#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools
from typing import Any

import tensorflow as tf

import zfit.z.numpy as _znp

from ..settings import ztypes
from ..util.exception import BreakingAPIChangeError
from .tools import _auto_upcast


def exp(x):
    return _auto_upcast(_znp.exp(x=x))


@functools.wraps(tf.convert_to_tensor)
def convert_to_tensor(value, dtype=ztypes.float):
    return _znp.asarray(value, dtype)


def random_normal(*_, **__):
    msg = "Use z.random.normal instead."
    raise BreakingAPIChangeError(msg)


def random_uniform(*_, **__):
    msg = "Use z.random.uniform instead."
    raise BreakingAPIChangeError(msg)


def random_poisson(*_, **__):
    msg = "Use z.random.poisson instead."
    raise BreakingAPIChangeError(msg)


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
        real_check = tf.debugging.check_numerics(tensor=_znp.real(tensor), message=message, name=name)
        imag_check = tf.debugging.check_numerics(tensor=_znp.imag(tensor), message=message, name=name)
        check_op = tf.group(real_check, imag_check)
    else:
        check_op = tf.debugging.check_numerics(tensor=tensor, message=message, name=name)
    return check_op


reduce_sum = _znp.sum

reduce_prod = _znp.prod

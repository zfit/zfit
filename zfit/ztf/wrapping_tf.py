import functools
from typing import Any

from tensorflow import DType
import tensorflow as tf

from .tools import _auto_upcast
from . import zextension
from ..settings import ztypes


def log(x, name=None):
    x = _auto_upcast(x)
    return tf.log(x=x, name=name)


def exp(x, name=None):
    x = _auto_upcast(x)
    return tf.exp(x=x, name=name)


@functools.wraps(tf.convert_to_tensor)
def convert_to_tensor(value, dtype=ztypes.float, name=None, preferred_dtype=ztypes.float):
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, preferred_dtype=preferred_dtype)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=ztypes.float, seed=None, name=None):
    return tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)


def random_uniform(shape, minval=0, maxval=None, dtype=ztypes.float, seed=None, name=None):
    return tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed, name=name)


def random_poisson(lam: Any, shape: Any, dtype: DType = ztypes.float, seed: Any = None, name: Any = None):
    return tf.random_poisson(lam=lam, shape=shape, dtype=dtype, seed=seed, name=name)


def square(x, name=None):
    x = _auto_upcast(x)
    return tf.square(x, name)


def sqrt(x, name=None):
    x = _auto_upcast(x)
    return tf.square(x, name=name)


#
# @functools.wraps(tf.reduce_sum)
# def reduce_sum(*args, **kwargs):
#     return tf.reduce_sum(*args, **kwargs)
reduce_sum = tf.reduce_sum

reduce_prod = tf.reduce_prod

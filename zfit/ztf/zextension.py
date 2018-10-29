import math as _mt

try:
    from math import inf as _inf
except ImportError:  # py34 remove try-except
    _inf = float('inf')

import tensorflow as tf
from zfit.settings import types as ztypes

inf = tf.constant(_inf, dtype=ztypes.float)


def constant(x, dtype=ztypes.float):
    return tf.constant(x, dtype)


pi = constant(_mt.pi)


def to_complex(number, dtype=ztypes.complex):
    return tf.cast(number, dtype=dtype)


def to_real(x, dtype=ztypes.float):
    return tf.cast(x, dtype=dtype)


def abs_square(x):
    return tf.real(x * tf.conj(x))


def nth_pow(x, n, name=None):
    """Calculate the nth power of the complex Tensor x.

    Args:
        x (tf.Tensor, complex):
        n (int >= 0): Power
        name (str): No effect, for API compatibility with tf.pow
    """
    if not n >= 0:
        raise ValueError("n (power) has to be >= 0. Currently, n={}".format(n))

    power = to_complex(1.)
    for _ in range(n):
        power *= x
    return power


def log(x, name=None):
    return tf.log(x=x, name=name)


def exp(x, name=None):
    return tf.exp(x=x, name=name)


# random sampling
def random_normal(shape, mean=0.0, stddev=1.0, dtype=ztypes.float, seed=None, name=None):
    return tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)


def random_uniform(shape, minval=0, maxval=None, dtype=ztypes.float, seed=None, name=None):
    return tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed, name=name)


# reduce functions

from tensorflow import reduce_prod, reduce_sum, reduce_mean

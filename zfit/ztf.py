import math as mt

import numpy as _znp
import tensorflow as _tf

from zfit.settings import types as _ztypes  # pay attention with the names in here!

# fill the following in to the namespace for (future) wrapping

# doesn't work...
# from tensorflow import *  # Yes, this is wanted. Yields an equivalent ztf BUT we COULD wrap it :)
try:
    from math import inf as _inf
except ImportError:  # py34 remove try-except
    _inf = float('inf')
inf = _tf.constant(_inf, dtype=_ztypes.float)


def constant(x, dtype=_ztypes.float):
    return _tf.constant(x, dtype)


pi = constant(mt.pi)


def to_complex(number, dtype=_ztypes.complex):
    return _tf.cast(number, dtype=dtype)


def to_real(x, dtype=_ztypes.float):
    return _tf.cast(x, dtype=dtype)


def abs_square(x):
    return _tf.real(x * _tf.conj(x))


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


# same as in TensorFlow, wrapped

def log(x, name=None):
    return _tf.log(x=x, name=name)


def exp(x, name=None):
    return _tf.exp(x=x, name=name)

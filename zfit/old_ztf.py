import math as _mt
import warnings

import tensorflow

import tensorflow as tf
from typing import Any

from .settings import ztypes as _ztypes  # pay attention with the names in here!

# fill the following in to the namespace for (future) wrapping

# doesn't work...
# from tensorflow import *  # Yes, this is wanted. Yields an equivalent ztf BUT we COULD wrap it :)

module_dict = tensorflow.__dict__
try:
    to_import = tensorflow.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]

imported = {}
failed_imports = []
for name in to_import:
    try:
        imported[name] = module_dict[name]
    except KeyError as error:
        failed_imports.append(name)
if failed_imports:
    warnings.warn("The following modules/attributes from TensorFlow could NOT be imported:\n{}".format(failed_imports))
globals().update(imported)

try:
    from math import inf as _inf
except ImportError:  # py34 remove try-except
    _inf = float('inf')
inf = tf.constant(_inf, dtype=_ztypes.float)


def constant(x, dtype=_ztypes.float):
    return tf.constant(x, dtype)


pi = constant(_mt.pi)


def to_complex(number, dtype=_ztypes.complex):
    return tf.cast(number, dtype=dtype)


def to_real(x, dtype=_ztypes.float):
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


def unstack_x(value: Any, num: Any = None, axis: int = 0, name: str = "unstack_x"):
    return ztf.unstack_x(value=value, num=num, axis=axis, name=name)


# same as in TensorFlow, wrapped

def log(x, name=None):
    return tf.log(x=x, name=name)


log.__doc__ = tf.log.__doc__


def exp(x, name=None):
    return tf.exp(x=x, name=name)


exp.__doc__ = tf.exp.__doc__

#  Copyright (c) 2019 zfit

import math as _mt

from typing import Any, Callable

import numpy as np
from math import inf as _inf

import tensorflow as tf
from ..settings import ztypes

inf = tf.constant(_inf, dtype=ztypes.float)


def constant(value, dtype=ztypes.float, shape=None, name="Const", verify_shape=False):
    return tf.constant(value, dtype=dtype, shape=shape, name=name, verify_shape=verify_shape)


pi = np.float64(_mt.pi)


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


def unstack_x(value: Any, num: Any = None, axis: int = -1, always_list: bool = False, name: str = "unstack_x"):
    """Unstack a Data object and return a list of (or a single) tensors in the right order.

    Args:
        value ():
        num (Union[]):
        axis (int):
        always_list (bool): If True, also return a list if only one element.
        name (str):

    Returns:
        Union[List[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor, None]:
    """
    if isinstance(value, list):
        if len(value) == 1 and not always_list:
            value = value[0]
        return value
    try:
        return value.unstack_x(always_list=always_list)
    except AttributeError:
        unstacked_x = tf.unstack(value=value, num=num, axis=axis, name=name)
    if len(unstacked_x) == 1 and not always_list:
        unstacked_x = unstacked_x[0]
    return unstacked_x


def stack_x(values, axis: int = -1, name: str = "stack_x"):
    return tf.stack(values=values, axis=axis, name=name)


# random sampling


def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, preferred_dtype=preferred_dtype)


def safe_where(condition: tf.Tensor, func: Callable, safe_func: Callable, values: tf.Tensor,
               value_safer: Callable = tf.ones_like) -> tf.Tensor:
    """Like :py:func:`tf.where` but fixes gradient `NaN` if func produces `NaN` with certain `values`.

    Args:
        condition (:py:class:`tf.Tensor`): Same argument as to :py:func:`tf.where`, a boolean :py:class:`tf.Tensor`
        func (Callable): Function taking `values` as argument and returning the tensor _in case
            condition is True_. Equivalent `x` of :py:func:`tf.where` but as function.
        safe_func (Callable): Function taking `values` as argument and returning the tensor
            _in case the condition is False_, Equivalent `y` of :py:func:`tf.where` but as function.
        values (:py:class:`tf.Tensor`): Values to be evaluated either by `func` or `safe_func` depending on
            `condition`.
        value_safer (Callable): Function taking `values` as arguments and returns "safe" values
            that won't cause troubles when given to`func` or by taking the gradient with respect
            to `func(value_safer(values))`.

    Returns:
        :py:class:`tf.Tensor`:
    """
    safe_x = tf.where(condition=condition, x=values, y=value_safer(values))
    result = tf.where(condition=condition, x=func(safe_x), y=safe_func(values))
    return result


def run_no_nan(func, x):
    from zfit.core.data import Data

    value_with_nans = func(x=x)
    if value_with_nans.dtype in (tf.complex128, tf.complex64):
        value_with_nans = tf.real(value_with_nans) + tf.imag(value_with_nans)  # we care only about NaN or not
    finite_bools = tf.debugging.is_finite(tf.cast(value_with_nans, dtype=tf.float64))
    finite_indices = tf.where(finite_bools)
    new_x = tf.gather_nd(params=x, indices=finite_indices)
    new_x = Data.from_tensor(obs=x.obs, tensor=new_x)
    vals_no_nan = func(x=new_x)
    result = tf.scatter_nd(indices=finite_indices, updates=vals_no_nan,
                           shape=tf.shape(value_with_nans, out_type=finite_indices.dtype))
    return result

# reduce functions

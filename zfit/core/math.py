#  Copyright (c) 2019 zfit

import itertools

import numdifftools
import tensorflow as tf
from typing import Iterable, Callable

from zfit.util.container import convert_to_container
from .. import z
from .parameter import Parameter
from ..settings import ztypes


def poly_complex(*args, real_x=False):
    """Complex polynomial with the last arg being x.

    Args:
        *args (tf.Tensor or equ.): Coefficients of the polynomial
        real_x (bool): If True, x is assumed to be real.

    Returns:
        tf.Tensor:
    """

    args = list(args)
    x = args.pop()
    if real_x is not None:
        pow_func = tf.pow
    else:
        pow_func = z.nth_pow
    return tf.add_n([coef * z.to_complex(pow_func(x, p)) for p, coef in enumerate(args)])


def interpolate(t, c):
    """Multilinear interpolation on a rectangular grid of arbitrary number of dimensions.

    Args:
        t (tf.Tensor): Grid (of rank N)
        c (tf.Tensor): Tensor of coordinates for which the interpolation is performed

    Returns:
        tf.Tensor: 1D tensor of interpolated value
    """
    rank = len(t.get_shape())
    ind = tf.cast(tf.floor(c), tf.int32)
    t2 = tf.pad(tensor=t, paddings=rank * [[1, 1]], mode='SYMMETRIC')
    wts = []
    for vertex in itertools.product([0, 1], repeat=rank):
        ind2 = ind + tf.constant(vertex, dtype=tf.int32)
        weight = tf.reduce_prod(input_tensor=1. - tf.abs(c - tf.cast(ind2, dtype=ztypes.float)), axis=1)
        wt = tf.gather_nd(t2, ind2 + 1)
        wts += [weight * wt]
    interp = tf.reduce_sum(input_tensor=tf.stack(wts), axis=0)
    return interp


def numerical_hessian():
    pass


def numerical_gradient(func: Callable, params: Iterable[Parameter]) -> tf.Tensor:
    """

    Returns:
        Tensor: Gradients
    """
    params = convert_to_container(params)

    def wrapped_func(param_values):
        for param, value in zip(params, param_values):
            param.assign(value)
        return func().numpy()

    param_vals = tf.stack(params)
    original_vals = [param.read_value() for param in params]
    grad_func = numdifftools.Gradient(wrapped_func)
    gradients = tf.py_function(grad_func, inp=[param_vals],
                               Tout=tf.float64)
    if gradients.shape == ():
        gradients = tf.reshape(gradients, shape=(1,))
    gradients.set_shape((len(param_vals),))
    for param, val in zip(params, original_vals):
        param.set_value(val)
    return gradients


def automatic_gradient(func: Callable, params: Iterable[Parameter]) -> tf.Tensor:
    """

    Returns:
        Tensor: Gradients
    """

    with tf.GradientTape(persistent=False) as tape:
        value = func()

    gradients = tape.gradient(value, sources=params)
    return gradients


if __name__ == '__main__':
    import zfit

    param1 = zfit.Parameter('param1', 4.)


    def func1():
        return param1 ** 2

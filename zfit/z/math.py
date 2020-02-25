#  Copyright (c) 2020 zfit

import itertools
from typing import Iterable, Callable

import numdifftools
import tensorflow as tf

from ..settings import ztypes
from ..util.container import convert_to_container


def poly_complex(*args, real_x=False):
    """Complex polynomial with the last arg being x.

    Args:
        *args (tf.Tensor or equ.): Coefficients of the polynomial
        real_x (bool): If True, x is assumed to be real.

    Returns:
        tf.Tensor:
    """
    from .. import z

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


def numerical_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> tf.Tensor:
    """Calculate numerically the gradients of func() with respect to `params`.

    Args:
        func (Callable): Function without arguments that depends on `params`
        params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        `tf.Tensor`: gradients
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


def numerical_value_gradients(func: Callable, params: Iterable["zfit.Parameter"]) -> [tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients of `func()` with respect to `params`, also returns the value of `func()`.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            tuple(`tf.Tensor`, `tf.Tensor`): value, gradient
    """
    return func(), numerical_gradient(func, params)


def numerical_hessian(func: Callable, params: Iterable["zfit.Parameter"], hessian=None) -> tf.Tensor:
    """Calculate numerically the hessian matrix of func with respect to `params`.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            `tf.Tensor`: hessian matrix
    """
    params = convert_to_container(params)

    def wrapped_func(param_values):
        for param, value in zip(params, param_values):
            param.assign(value)
        return func().numpy()

    param_vals = tf.stack(params)
    original_vals = [param.read_value() for param in params]

    if hessian == 'diag':
        hesse_func = numdifftools.Hessdiag(wrapped_func)
    else:
        hesse_func = numdifftools.Hessian(wrapped_func)
    computed_hessian = tf.py_function(hesse_func, inp=[param_vals],
                                      Tout=tf.float64)

    if hessian == 'diag':
        computed_hessian.set_shape((len(param_vals),))
    else:
        computed_hessian.set_shape((len(param_vals), len(param_vals)))

    for param, val in zip(params, original_vals):
        param.set_value(val)
    return computed_hessian


def numerical_value_gradients_hessian(func: Callable, params: Iterable["zfit.Parameter"], hessian=None) -> [tf.Tensor,
                                                                                                            tf.Tensor,
                                                                                                            tf.Tensor]:
    """Calculate numerically the gradients and hessian matrix of `func()` wrt `params`; also return `func()`.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            tuple(`tf.Tensor`, `tf.Tensor`, `tf.Tensor`): value, gradient and hessian matrix
    """
    value, gradients = numerical_value_gradients(func, params)
    hessian = numerical_hessian(func, params, hessian=hessian)

    return value, gradients, hessian


def autodiff_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> tf.Tensor:
    """Calculate using autodiff the gradients of `func()` wrt `params`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            `tf.Tensor`: gradient
    """
    return autodiff_value_gradients(func, params)[1]


def autodiff_value_gradients(func: Callable, params: Iterable["zfit.Parameter"]) -> [tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients of `func()` wrt `params`; also return `func()`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            tuple(`tf.Tensor`, `tf.Tensor`): value and gradient
    """
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        tape.watch(params)
        value = func()
    gradients = tape.gradient(value, sources=params)
    return value, gradients


def autodiff_hessian(func: Callable, params: Iterable["zfit.Parameter"], hessian=None) -> tf.Tensor:
    """Calculate using autodiff the hessian matrix of `func()` wrt `params`.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            `tf.Tensor`: hessian matrix
    """

    return automatic_value_gradients_hessian(func, params, hessian=hessian)[2]


def automatic_value_gradients_hessian(func: Callable, params: Iterable["zfit.Parameter"], hessian=None) -> [tf.Tensor,
                                                                                                            tf.Tensor,
                                                                                                            tf.Tensor]:
    """Calculate using autodiff the gradients and hessian matrix of `func()` wrt `params`; also return `func()`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func (Callable): Function without arguments that depends on `params`
            params (ZfitParameter): Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            tuple(`tf.Tensor`, `tf.Tensor`, `tf.Tensor`): value, gradient and hessian matrix
    """
    from .. import z
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(params)
        loss, gradients = autodiff_value_gradients(func=func, params=params)
        if hessian != 'diag':
            gradients_tf = z.convert_to_tensor(gradients)
    if hessian == 'diag':
        computed_hessian = z.convert_to_tensor(
            [tape.gradient(grad, sources=param) for param, grad in zip(params, gradients)])
    else:
        computed_hessian = z.convert_to_tensor(tape.jacobian(gradients_tf, sources=params))
    return loss, gradients, computed_hessian

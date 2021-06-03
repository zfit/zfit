#  Copyright (c) 2021 zfit

from typing import Callable, Iterable, Optional

import numdifftools
import tensorflow as tf

import zfit.z.numpy as znp

from ..util.container import convert_to_container
from ..util.deprecation import deprecated
from .tools import _auto_upcast
from .zextension import convert_to_tensor


def poly_complex(*args, real_x=False):
    """Complex polynomial with the last arg being x.

    Args:
        *args: Coefficients of the polynomial
        real_x: If True, x is assumed to be real.

    Returns:
    """
    from .. import z

    args = list(args)
    x = args.pop()
    if real_x is not None:
        pow_func = znp.power
    else:
        pow_func = z.nth_pow
    return tf.add_n([coef * z.to_complex(pow_func(x, p)) for p, coef in enumerate(args)])


def numerical_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> tf.Tensor:
    """Calculate numerically the gradient of func() with respect to `params`.

    Args:
        func: Function without arguments that depends on `params`
        params: Parameters that `func` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Gradients
    """
    from ..core.parameter import assign_values

    params = convert_to_container(params)

    def wrapped_func(param_values):
        assign_values(params, param_values)
        value = func()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return value

    param_vals = znp.stack(params)
    original_vals = [param.value() for param in params]
    grad_func = numdifftools.Gradient(wrapped_func, order=2, base_step=1e-4)
    if tf.executing_eagerly():
        grad_vals = grad_func(param_vals)
        gradient = convert_to_tensor(grad_vals)
    else:
        gradient = tf.numpy_function(grad_func, inp=[param_vals],
                                     Tout=tf.float64)
    if gradient.shape == ():
        gradient = znp.reshape(gradient, newshape=(1,))
    gradient.set_shape(param_vals.shape)
    assign_values(params, original_vals)
    return gradient


def numerical_value_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> [tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients of `func()` with respect to `params`, also returns the value of `func()`.

    Args:
        func: Function without arguments that depends on `params`
        params: Parameters that `func` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Value, gradient
    """
    return func(), numerical_gradient(func, params)


deprecated(None, "Use `numerical_value_gradient` instead.")


def numerical_value_gradients(*args, **kwargs):
    return numerical_value_gradients(*args, **kwargs)


def numerical_hessian(func: Optional[Callable],
                      params: Iterable["zfit.Parameter"],
                      hessian=None) -> tf.Tensor:
    """Calculate numerically the hessian matrix of func with respect to `params`.

    Args:
        func: Function without arguments that depends on `params`
        params: Parameters that `func` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Hessian matrix
    """
    from ..core.parameter import assign_values

    params = convert_to_container(params)

    def wrapped_func(param_values):
        assign_values(params, param_values)
        value = func()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return value

    param_vals = znp.stack(params)
    original_vals = [param.value() for param in params]

    if hessian == 'diag':
        hesse_func = numdifftools.Hessdiag(wrapped_func,
                                           order=2,
                                           # TODO: maybe add step to remove numerical problems?
                                           base_step=1e-4
                                           )
    else:
        hesse_func = numdifftools.Hessian(wrapped_func,
                                          order=2,
                                          base_step=1e-4,
                                          )
    if tf.executing_eagerly():
        computed_hessian = convert_to_tensor(hesse_func(param_vals))
    else:
        computed_hessian = tf.numpy_function(hesse_func, inp=[param_vals],
                                             Tout=tf.float64)
    n_params = param_vals.shape[0]
    if hessian == 'diag':
        computed_hessian.set_shape((n_params,))
    else:
        computed_hessian.set_shape((n_params, n_params))

    assign_values(params, original_vals)
    return computed_hessian


def numerical_value_gradient_hessian(func: Optional[Callable],
                                     params: Iterable["zfit.Parameter"],
                                     gradient: Optional[Callable] = None,
                                     hessian: Optional[str] = None) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients and hessian matrix of `func()` wrt `params`; also return `func()`.

    Args:
        func: Function without arguments that depends on `params`
        params: Parameters that `func` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Value, gradient and hessian matrix
    """
    if params is None:
        raise ValueError("params cannot be None")
    if func is None and gradient is None:
        raise ValueError("Either func or grad has to be given")
    value, gradients = numerical_value_gradient(func, params)
    hessian = numerical_hessian(func, params, hessian=hessian)

    return value, gradients, hessian


@deprecated(None, "Use `numerical_value_gradient_hessian` instead.")
def numerical_value_gradients_hessian(*args, **kwargs):
    return numerical_value_gradient_hessian(*args, **kwargs)


def autodiff_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> tf.Tensor:
    """Calculate using autodiff the gradients of `func()` wrt `params`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func: Function without arguments that depends on `params`
            params: Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Gradient
    """
    return autodiff_value_gradients(func, params)[1]


def autodiff_value_gradient(func: Callable, params: Iterable["zfit.Parameter"]) -> [tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients of `func()` wrt `params`; also return `func()`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func: Function without arguments that depends on `params`
            params: Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value and gradient
    """
    with tf.GradientTape(persistent=False,  # needs to be persistent for a call from hessian.
                         watch_accessed_variables=False) as tape:
        tape.watch(params)
        value = func()
    gradients = tape.gradient(value, sources=params)
    return value, gradients


@deprecated(None, "Use `autodiff_value_gradient` instead.")
def autodiff_value_gradients(*args, **kwargs):
    return autodiff_value_gradient(*args, **kwargs)


def autodiff_hessian(func: Callable, params: Iterable["zfit.Parameter"], hessian=None) -> tf.Tensor:
    """Calculate using autodiff the hessian matrix of `func()` wrt `params`.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func: Function without arguments that depends on `params`
            params: Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Hessian matrix
    """

    return automatic_value_gradients_hessian(func, params, hessian=hessian)[2]


def automatic_value_gradient_hessian(func: Callable = None, params: Iterable["zfit.Parameter"] = None,
                                     value_grad_func=None,
                                     hessian=None) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients and hessian matrix of `func()` wrt `params`; also return `func()`.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using `tf.*` operations only can use this technique.

        Args:
            func: Function without arguments that depends on `params`
            params: Parameters that `func` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value, gradient and hessian matrix
    """
    if params is None:
        raise ValueError("Parameters have to be specified, are currently None.")
    if func is None and value_grad_func is None:
        ValueError("Either `func` or `value_grad_func` has to be specified.")

    from .. import z
    persistant = hessian == 'diag' or tf.executing_eagerly()  # currently needed, TODO: can we better parallelize that?
    with tf.GradientTape(persistent=persistant, watch_accessed_variables=False) as tape:
        tape.watch(params)
        if callable(value_grad_func):
            loss, gradients = value_grad_func(params)
        else:
            loss, gradients = autodiff_value_gradients(func=func, params=params)
        if hessian != 'diag':
            gradients_tf = znp.stack(gradients)
    if hessian == 'diag':
        computed_hessian = znp.stack(
            [tape.gradient(grad, sources=param) for param, grad in zip(params, gradients)]
        )
    else:
        computed_hessian = z.convert_to_tensor(tape.jacobian(gradients_tf, sources=params,
                                                             experimental_use_pfor=False  # causes TF bug? Slow..
                                                             ))
    del tape
    return loss, gradients, computed_hessian


@deprecated(None, "Use `automatic_value_gradient_hessian` instead.")
def automatic_value_gradients_hessian(*args, **kwargs):
    return automatic_value_gradient_hessian(*args, **kwargs)

@tf.function(autograph=False)
def reduce_geometric_mean(input_tensor, axis=None, keepdims=False):
    log_mean = znp.mean(znp.log(input_tensor), axis=axis, keepdims=keepdims)
    return znp.exp(log_mean)


def log(x):
    x = _auto_upcast(x)
    return _auto_upcast(znp.log(x=x))

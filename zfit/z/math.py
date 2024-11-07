#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..util.exception import BreakingAPIChangeError

if TYPE_CHECKING:
    import zfit

from collections.abc import Callable, Iterable

import numdifftools
import tensorflow as tf
import tensorflow_probability as tfp

from ..util.container import convert_to_container
from ..util.deprecation import deprecated
from . import numpy as znp
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
    pow_func = znp.power if real_x is not None else z.nth_pow
    return tf.add_n([coef * z.to_complex(pow_func(x, p)) for p, coef in enumerate(args)])


def numerical_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tf.Tensor:
    """Calculate numerically the gradient of func() with respect to ``params``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Gradients
    """
    from ..core.parameter import assign_values

    params = convert_to_container(params)

    def wrapped_func(param_values):
        assign_values(params, param_values)
        return func()

    param_vals = znp.array(params)
    param_vals = znp.atleast_1d(param_vals)

    # fails for apple silicon runners weirdly, but preferable
    # def grad_func(values):
    #     # todo: adjust rtol?
    #     gradients = jacobi.jacobi(wrapped_func, values)[0]
    #     gradients = znp.asarray(gradients)
    #     gradients.set_shape(values.shape)
    #     return gradients  # element 1 are the errors

    grad_func = numdifftools.Gradient(wrapped_func, order=2, base_step=1e-1)
    gradient = tf.numpy_function(grad_func, inp=[param_vals], Tout=tf.float64)
    gradient = znp.atleast_1d(gradient)
    gradient.set_shape(param_vals.shape)
    with tf.control_dependencies([gradient]):
        assign_values(params, param_vals)
        return gradient


def numerical_value_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> [tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients of ``func()`` with respect to ``params``, also returns the value of
    ``func()``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Value, gradient
    """
    return func(), numerical_gradient(func, params)


deprecated(None, "Use `numerical_value_gradient` instead.")


def numerical_value_gradients(*args, **kwargs):
    return numerical_value_gradients(*args, **kwargs)


def numerical_hessian(func: Callable | None, params: Iterable[zfit.Parameter], hessian=None) -> tf.Tensor:
    """Calculate numerically the hessian matrix of func with respect to ``params``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Hessian matrix
    """
    from ..core.parameter import assign_values

    params = convert_to_container(params)

    def wrapped_func(param_values):
        assign_values(params, param_values)
        value = func()
        return np.asarray(value)  # numdifftools doesn't understand the TF dtype

    nparams = len(params)
    param_vals = znp.array(params)

    if hessian == "diag":
        hesse_func = numdifftools.Hessdiag(
            wrapped_func,
            order=3,
            # TODO: maybe add step to remove numerical problems?
            base_step=1e-1,
        )
    else:
        hesse_func = numdifftools.Hessian(
            wrapped_func,
            order=3,
            base_step=1e-1,
        )
    if tf.executing_eagerly():
        computed_hessian = convert_to_tensor(hesse_func(param_vals))
    else:
        computed_hessian = tf.numpy_function(hesse_func, inp=[param_vals], Tout=tf.float64)
    if hessian == "diag":
        computed_hessian.set_shape((nparams,))
    else:
        computed_hessian.set_shape((nparams, nparams))

    with tf.control_dependencies([computed_hessian]):
        assign_values(params, param_vals)
        return computed_hessian


def numerical_value_gradient_hessian(
    func: Callable | None,
    params: Iterable[zfit.Parameter],
    gradient: Callable | None = None,
    hessian: str | None = None,
) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients and hessian matrix of ``func()`` wrt ``params``; also return ``func()``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Value, gradient and hessian matrix
    """
    if params is None:
        msg = "params cannot be None"
        raise ValueError(msg)
    if func is None and gradient is None:
        msg = "Either func or grad has to be given"
        raise ValueError(msg)
    value, gradients = numerical_value_gradient(func, params)
    hessian = numerical_hessian(func, params, hessian=hessian)

    return value, gradients, hessian


@deprecated(None, "Use `numerical_value_gradient_hessian` instead.")
def numerical_value_gradients_hessian(*args, **kwargs):
    return numerical_value_gradient_hessian(*args, **kwargs)


def autodiff_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tf.Tensor:
    """Calculate using autodiff the gradients of ``func()`` wrt ``params``.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Gradient
    """
    return autodiff_value_gradient(func, params)[1]


def _extract_tfparams(
    params: Iterable[zfit.Parameter] | zfit.Parameter,
) -> list[tf.Variable]:
    """Extract the tf.Variable from the parameters.

    Args:
        params:

    Returns:
        tf.Variable
    """
    return params
    # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
    # import zfit
    # params = convert_to_container(params)
    # tf_params = []
    # for param in params:
    #     if isinstance(param, tf.Variable):
    #
    #         # TODO: reactivate if WrappedVariables are used
    #         # if isinstance(param, zfit.Parameter):
    #         #     raise ValueError("The parameter cannot be a tf.Variable and a zfit.Parameter at the same time.")
    #         variable = param
    #     else:
    #         if not isinstance(param, zfit.Parameter):
    #             raise ValueError("The parameter has to be either a tf.Variable or a zfit.Parameter.")
    #         variable = param.variable
    #     tf_params.append(variable)
    # return tf_params


def autodiff_value_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> [tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients of ``func()`` wrt ``params``; also return ``func()``.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value and gradient
    """
    # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
    # params = _extract_tfparams(params)
    with tf.GradientTape(
        persistent=False,  # needs to be persistent for a call from hessian.
        watch_accessed_variables=False,
    ) as tape:
        tape.watch(params)
        value = func()
    gradients = tape.gradient(value, sources=params)
    gradients = znp.asarray(gradients)
    return value, gradients


def autodiff_value_gradients(*args, **kwargs):  # noqa: ARG001
    msg = "Use `autodiff_value_gradient` instead."
    raise BreakingAPIChangeError(msg)


def autodiff_hessian(func: Callable, params: Iterable[zfit.Parameter], hessian=None) -> tf.Tensor:
    """Calculate using autodiff the hessian matrix of ``func()`` wrt ``params``.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Hessian matrix
    """

    return automatic_value_gradient_hessian(func, params, hessian=hessian)[2]


def automatic_value_gradient_hessian(
    func: Callable | None = None,
    params: Iterable[zfit.Parameter] | None = None,
    value_grad_func=None,
    hessian=None,
) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients and hessian matrix of ``func()`` wrt ``params``; also return ``func()``.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value, gradient and hessian matrix
    """
    if params is None:
        msg = "Parameters have to be specified, are currently None."
        raise ValueError(msg)
    if func is None and value_grad_func is None:
        msg = "Either `func` or `value_grad_func` has to be specified."
        raise ValueError(msg)

    from .. import z

    # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
    # params = _extract_tfparams(params)
    # persistant is needed for jacobian later, which fails otherwise if we try again without the vectorization
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(params)
        if callable(value_grad_func):
            loss, gradients = value_grad_func(params)
        else:
            loss, gradients = autodiff_value_gradient(func=func, params=params)
        if hessian == "diag":
            gradients = tf.unstack(gradients)
            # gradients_tf = znp.stack(gradients)
    if hessian == "diag":
        computed_hessian = znp.stack([tape.gradient(grad, sources=param) for param, grad in zip(params, gradients)])
        # gradfunc = lambda par_grad: tape.gradient(par_grad[0], sources=par_grad[1])
        # computed_hessian = tf.vectorized_map(gradfunc, zip(params, gradients))
    else:
        for usepfor in [True, False]:
            try:
                computed_hessian = z.convert_to_tensor(
                    tape.jacobian(
                        gradients,
                        sources=params,
                        experimental_use_pfor=usepfor,  # causes TF bug? Slow..
                    )
                )
            except ValueError as error:
                if "Encountered an exception while vectorizing the jacobian computation." in str(error):
                    continue
                raise
            else:
                break
    del tape
    return loss, gradients, computed_hessian


def automatic_value_gradients_hessian(*args, **kwargs):  # noqa: ARG001
    msg = "Use `automatic_value_gradient_hessian` instead."
    raise BreakingAPIChangeError(msg)


# @z.function  # TODO: circular import, improve?
def reduce_geometric_mean(input_tensor, axis=None, weights=None, keepdims=False):
    if weights is not None:
        log_mean = tf.nn.weighted_moments(log(input_tensor), axes=axis, frequency_weights=weights)[0]
    else:
        log_mean = znp.mean(znp.log(input_tensor), axis=axis, keepdims=keepdims)
    return znp.exp(log_mean)


def log(x):
    x = _auto_upcast(x)
    return _auto_upcast(znp.log(x=x))


def weighted_quantile(x, quantiles, weights=None, side="middle"):
    """Very close to numpy.percentile, but supports weights.

    NOTE: quantiles should be in [0, 1]!
    :param x: tensor with data
    :param quantiles: array-like with many quantiles needed
    :param weights: array-like of the same length as ``x``
    :return: numpy.array with computed quantiles.
    """
    if weights is None:
        return tfp.stats.percentile(x, 100 * quantiles)
    x = znp.array(x)
    quantiles = znp.array(quantiles)
    quantiles = znp.reshape(quantiles, (-1,))
    weights = znp.array(weights)

    sorter = znp.argsort(x)
    x = tf.gather(x, sorter)
    weights = tf.gather(weights, sorter)

    weighted_quantiles = znp.cumsum(weights) - 0.5 * weights

    weighted_quantiles /= znp.sum(weights)
    if side == "middle":
        quantile_index_left = tf.searchsorted(weighted_quantiles, quantiles, side="left")
        quantile_index_right = tf.searchsorted(weighted_quantiles, quantiles, side="right")

        calculated_left = tf.gather(x, quantile_index_left)
        calculated_right = tf.gather(x, quantile_index_right)
        calculated = (calculated_left + calculated_right) / 2
    elif side in ("left", "right"):
        quantile_index = tf.searchsorted(weighted_quantiles, quantiles, side=side)

        calculated = tf.gather(x, quantile_index)
    return calculated

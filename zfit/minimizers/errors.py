#  Copyright (c) 2023 zfit

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jacobi

if TYPE_CHECKING:
    import zfit

from collections.abc import Callable

import logging
from functools import lru_cache, wraps

import numpy as np
import scipy.stats
import tensorflow as tf
from scipy import optimize

import zfit.z.numpy as znp

from .. import z
from ..core.interfaces import ZfitIndependentParameter
from ..core.parameter import assign_values, assign_values_jit
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_args


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""

    pass


class FailEvalLossNaN(Exception):
    pass


class RootFound(Exception):
    """Exception class for cases where a root is found, since SciPy root solvers don't really respect tol or xtol on
    initial evaluation."""

    pass


@deprecated_args(None, "Use cl for confidence level instead.", "sigma")
def compute_errors(
    result: zfit.result.FitResult,
    params: list[ZfitIndependentParameter],
    cl: float | None = None,
    rtol: float | None = None,
    method: str | None = None,
    covariance_method: str | Callable | None = None,
    sigma: float = None,
) -> tuple[
    dict[ZfitIndependentParameter, dict[str, float]],
    zfit.result.FitResult | None,
]:
    """Compute asymmetric errors of parameters by profiling the loss function in the fit result.

    This method finds the value for a given parameter where the loss function is ``cl`` away: for example
    for a cl of 68.3%, this is one (multiplied by the errordef). The other parameters are also minimized and
    not fixed. This method is comparably computationally intensive and, if possible, ``hesse`` should be used.
    However, since ``hesse`` does not capture asymmetric or non-parabolic shaped profiles well, this method is
    preferable.

    Args:
        result: fit result to be used to compute the uncertainties.
        params: The parameters to calculate the
            error. If None, use all parameters.
        cl: Confidence Level of the parameter to be determined. Defaults to 68.3%.
        rtol: relative tol between the computed and the exact roots
        method: type of solver, ``method`` argument of :py:func:`scipy.optimize.root`. Defaults to "hybr".
        covariance_method: The method to use to calculate the correlation matrix, will be forwarded directly
            to :py:meth:`FitResult.covariance`. Valid choices are
            by default {'minuit_hesse', 'hesse_np'} (or any other method defined in the result)
            or a Callable.

    Returns:
        out:
            A ``dict`` containing as keys the parameter and as value a ``dict`` which
            contains two keys 'lower' and 'upper', holding the calculated errors.
            Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        out: a fit result is returned when a new minimum is found during the loss scan
    """
    if rtol is None:
        rtol = 0.03
    method = "hybr" if method is None else method
    if cl is None:
        if sigma is None:
            sigma = 1

    else:
        if sigma is None:
            sigma = scipy.stats.chi2(1).ppf(cl) ** 0.5
        else:
            raise ValueError("Cannot specify both sigma and cl.")
    # TODO: how to name things, sigma or cl?

    params = convert_to_container(params)
    new_result = None

    all_params = list(result.params.keys())
    loss = result.loss
    errordef = loss.errordef
    fmin = result.fmin
    rtol *= errordef
    minimizer = result.minimizer

    covariance = result.covariance(method=covariance_method, as_dict=True)
    if covariance is None:
        covariance = result.covariance(method="hesse_np", as_dict=True)
    if covariance is None:
        raise RuntimeError(
            "Could not compute covariance matrix. Check if the minimum is valid."
        )
    param_errors = {param: covariance[(param, param)] ** 0.5 for param in all_params}
    param_scale = np.array(list(param_errors.values()))

    ncalls = 0
    loss_min_tol = minimizer.tol * errordef * 2  # 2 is just to be tolerant
    try:
        to_return = {}
        for param in params:
            assign_values(all_params, result)

            logging.info(f"profiling the parameter {param}")
            param_error = param_errors[param]
            param_value = result.params[param]["value"]

            initial_values = {"lower": [], "upper": []}
            direction = {"lower": -1, "upper": 1}

            for ap in all_params:
                ap_value = result.params[ap]["value"]
                error_factor = (
                    covariance[(param, ap)] * (2 * errordef / param_error**2) ** 0.5
                )
                for d in ["lower", "upper"]:
                    step = direction[d] * error_factor * sigma
                    for ntrial in range(50):
                        ap_value_init = ap_value + step
                        if ap_value_init < ap.lower or ap_value_init > ap.upper:
                            step *= 0.8
                        else:
                            break
                    else:
                        raise RuntimeError(
                            f"Could not find a valid initial value for {ap} in {d} direction after {ntrial + 1} trials."
                            f" step tried: {step}. This should not happes, the error probably looks weird. Maybe plot"
                            f" the loss function for different parameter values and check if it looks reasonable."
                        )

                    initial_values[d].append(ap_value_init)

            index_poi = all_params.index(param)  # remember the index
            _ = loss.value_gradient(
                params=all_params
            )  # to make sure the loss is compiled

            @z.function(wraps="gradient")
            def optimized_loss_gradient(values, index):
                assert isinstance(index, int)
                assign_values(all_params, values)
                loss_value, gradient = loss.value_gradient(params=all_params)
                if isinstance(gradient, (tuple, list)):
                    gradient = znp.asarray(gradient)
                gradient = znp.concatenate(
                    [gradient[:index_poi], gradient[index_poi + 1 :]]
                )
                return loss_value, gradient

            # TODO: improvement, use jacobian?
            root = None
            ntol = 999  # if it's right in the beginning, we think it's fine

            # TODO: should we add a "robust" or similar option to not skip this?
            # or evaluate and then decide ,maybe use krylov as it doesn't do a lot of calls in the beginning, it
            # approximates the jacobian

            def func(values, args):
                nonlocal ncalls, root, ntol
                ncalls += 1
                swap_sign = args

                try:
                    loss_value, gradient = optimized_loss_gradient(values, index_poi)
                except tf.errors.InvalidArgumentError:
                    msg = (
                        f"The evaluation of the errors of {param.name} failed due to too many NaNs"
                        " being produced in the loss and/or its gradient. This is most probably"
                        " caused by negative values returned from the PDF."
                    )
                    raise FailEvalLossNaN(msg)
                zeroed_loss = loss_value.numpy() - fmin

                gradient = znp.array(gradient)

                if swap_sign(param):  # mirror at x-axis to remove second zero
                    zeroed_loss = -zeroed_loss
                    gradient = -gradient
                    logging.info("Swapping sign in error calculation 'zfit_error'")

                elif zeroed_loss < -loss_min_tol:
                    assign_values(all_params, values)  # set values to the new minimum
                    raise NewMinimum("A new minimum is found.")

                downward_shift = errordef * sigma**2
                shifted_loss = zeroed_loss - downward_shift

                if abs(shifted_loss) < rtol:
                    if ntol > 3:
                        root = values[index_poi]
                        raise RootFound()
                    ntol += 1
                else:
                    ntol = 0

                return znp.concatenate([[shifted_loss], gradient])

            to_return[param] = {}
            swap_sign = {
                "lower": lambda p: p > param_value,
                "upper": lambda p: p < param_value,
            }
            for d in ["lower", "upper"]:
                try:
                    root_result = optimize.root(
                        fun=func,
                        args=(swap_sign[d],),
                        x0=np.array(initial_values[d]),
                        tol=rtol,  # we won't stop like this anyway
                        options={
                            "diag": 1 / param_scale,  # scale factor for variables
                        },
                        method=method,
                    )
                except RootFound:
                    assert root is not None, "Should be changed inside function."
                else:
                    warnings.warn(
                        f"The root finding did not converge below {rtol} but stopped by its own criteria."
                    )
                    root = root_result.x[index_poi]
                to_return[param][d] = root - param_value

        assign_values(all_params, result)

    except NewMinimum as e:
        from .. import settings

        if settings.get_verbosity() >= 5:
            print(e)
        minimizer = result.minimizer
        loss = result.loss
        new_found_fmin = loss.value()
        new_result = minimizer.minimize(loss=loss)
        if new_result.fmin >= new_found_fmin + loss_min_tol:
            raise RuntimeError(
                "A new minimum was discovered but the minimizer was not able to find this on himself. "
                "This behavior is currently an exception but will most likely change in the future."
            )
        to_return, new_result_ = compute_errors(
            result=new_result, params=params, cl=cl, rtol=rtol, method=method
        )
        if new_result_ is not None:
            new_result = new_result_
    return to_return, new_result


def numerical_pdf_jacobian(func, params):  # TODO: jit?
    params = list(params.values())
    return znp.asarray(jacobi.jacobi(func, params)[0].T)


# @z.function(wraps="autodiff")
def autodiff_pdf_jacobian(func, params):
    params = list(params.values())
    # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
    # params = z.math._extract_tfparams(params)

    # the below fails for some cases (i.e. CB) with an internal error
    # ValueError: Internal error: Tried to take gradients (or similar) of a variable without handle data:
    # Tensor("Placeholder_1:0", shape=(), dtype=resource)

    # we didn't report that yet, it's too hard to reproduce with a minimal example currently.

    # with tf.GradientTape(watch_accessed_variables=False) as t2:
    #     t2.watch(params)
    #     with tf.GradientTape(watch_accessed_variables=False) as t1:
    #         t1.watch(params)
    #         values = func()
    #
    #     grad = t1.gradient(values, params)
    #     grad = tf.convert_to_tensor(grad)
    # jacobian = t2.jacobian(grad, params)

    columns = []

    for p in params:
        vector = np.zeros(len(params))
        vector[params.index(p)] = 1.0
        with tf.autodiff.ForwardAccumulator(params, list(vector)) as acc:
            values = func()
        columns.append(acc.jvp(values))

    jacobian = z.convert_to_tensor(columns)

    return jacobian


def covariance_with_weights(method, result, params):
    """Compute the covariance matrix of the parameters with weights."""
    from .. import run

    run.assert_executing_eagerly()
    model = result.loss.model
    data = result.loss.data
    from zfit import run

    old_vals = run(params)

    Hinv_dict = method(result=result, params=params)  # inverse of the hessian matrix
    if not Hinv_dict:
        return {}
    Hinv = dict_to_matrix(params, Hinv_dict)

    def func():
        values = []
        for m, d in zip(model, data):
            v = m.log_pdf(d)
            if d.weights is not None:
                v *= d.weights
            values.append(v)
        return znp.concatenate(values, axis=0)

    params_dict = {p.name: p for p in params}

    if run.get_autograd_mode():
        try:
            jacobian = autodiff_pdf_jacobian(func=func, params=params_dict)
        except ValueError as error:
            raise ValueError(
                "The autodiff could not be performed for the jacobian matrix. This can have a natural cause (see error above)"
                " or be a bug in the autodiff. In the latter case, you can try to use the numerical jacobian instead using:\n"
                "with zfit.run.set_autograd_mode(False):\n"
                "    # calculate here, i.e. result.errors()"
            ) from error
    else:

        def wrapped_func(values):
            assign_values(list(params_dict.values()), values)
            return np.array(func())

        jacobian = numerical_pdf_jacobian(func=wrapped_func, params=params_dict)

    C = znp.matmul(jacobian, znp.transpose(jacobian))
    covariance = np.asarray(znp.matmul(Hinv, znp.matmul(C, Hinv)))
    assign_values(params, old_vals)
    return matrix_to_dict(params, covariance)


def dict_to_matrix(params, matrix_dict):
    nparams = len(params)
    matrix = np.empty((nparams, nparams))
    if not matrix_dict:
        return None

    for i in range(nparams):
        pi = params[i]
        for j in range(i, nparams):
            pj = params[j]
            matrix[i, j] = matrix_dict[(pi, pj)]
            if i != j:
                matrix[j, i] = matrix_dict[(pi, pj)]

    return matrix


def matrix_to_dict(params, matrix):
    nparams = len(params)
    matrix_dict = {}
    if matrix is None:
        return matrix_dict

    for i in range(nparams):
        pi = params[i]
        for j in range(i, nparams):
            pj = params[j]
            matrix_dict[(pi, pj)] = matrix[i, j]
            if i != j:
                matrix_dict[(pj, pi)] = matrix[i, j]

    return matrix_dict


def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array.

    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = array_to_tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            """Iterates recursively."""
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator

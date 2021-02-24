#  Copyright (c) 2021 zfit
import logging
import time
from functools import wraps, lru_cache
from typing import Tuple, Dict, Optional, Callable, Union, List

import numdifftools
import numpy as np
import scipy.stats
import tensorflow as tf
from scipy import optimize

from .. import z, settings
from ..core.interfaces import ZfitIndependentParameter
from ..param import set_values
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_args


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""
    pass


class FailEvalLossNaN(Exception):
    pass


@deprecated_args(None, "Use cl for confidence level instead.", 'sigma')
def compute_errors(result: "zfit.minimizers.fitresult.FitResult",
                   params: List[ZfitIndependentParameter],
                   cl: float = None,
                   sigma: float = 1,
                   rtol: float = 0.005,
                   method: Optional[str] = None,
                   covariance_method: Optional[Union[str, Callable]] = None
                   ) -> Tuple[Dict[ZfitIndependentParameter, Dict[str, float]],
                              Union["zfit.result.FitResult", None]]:
    """
    Computes asymmetric errors of parameters by profiling the loss function in the fit result.

    This method finds the value for a given parameter where the loss function is `cl` away: for example
    for a cl of 68.3%, this is one (multiplied by the errordef). The other parameters are also minimized and
    not fixed. This method is comparably computationally intensive and, if possible, `hesse` should be used.
    However, since `hesse` does not capture asymetric or non-parabolic shaped profiles well, this method is
    preferable.

    Args:
        result: fit result to be used to compute the uncertainties.
        params: The parameters to calculate the
            errors error. If None, use all parameters.
        cl: Confidence Level of the parameter to be determined. Defaults to 68.3%.
        sigma: Errors are calculated with respect to `sigma` std deviations.
        rtol: relative tol between the computed and the exact roots
        method: type of solver, `method` argument of :py:func:`scipy.optimize.root`. Defaults to "hybr".
        covariance_method: The method to use to calculate the correlation matrix, will be forwarded directly
            to :py:meth:`FitResult.covariance`. Valid choices are
            by default {'minuit_hesse', 'hesse_np'} (or any other method defined in the result)
            or a Callable.

    Returns:
        out:
            A `dict` containing as keys the parameter and as value a `dict` which
            contains two keys 'lower' and 'upper', holding the calculated errors.
            Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        out: a fit result is returned when a new minimum is found during the loss scan
    """
    method = "hybr" if method is None else method
    if cl is None:
        factor = 1.0
    else:
        factor = scipy.stats.chi2(1).ppf(cl)
    params = convert_to_container(params)
    new_result = None

    all_params = list(result.params.keys())
    loss = result.loss
    errordef = loss.errordef * factor
    fmin = result.fmin
    rtol *= errordef
    minimizer = result.minimizer

    covariance = result.covariance(method=covariance_method, as_dict=True)
    set_values(all_params, result)
    param_errors = {param: covariance[(param, param)] ** 0.5 for param in params}
    param_scale = np.array(list(param_errors.values()))

    ncalls = 0
    try:
        # start = time.time()
        to_return = {}
        for param in params:
            logging.info(f"profiling the parameter {param}")
            param_error = param_errors[param]
            param_value = result.params[param]["value"]
            other_params = [p for p in all_params if p != param]

            initial_values = {"lower": [], "upper": []}
            direction = {"lower": -sigma, "upper": sigma}

            for ap in all_params:
                ap_value = result.params[ap]["value"]
                error_factor = covariance[(param, ap)] * (2 * errordef / param_error ** 2) ** 0.5
                for d in ["lower", "upper"]:
                    ap_value += direction[d] * error_factor
                    initial_values[d].append(ap_value)

            # TODO: improvement, use jacobian?
            # @np_cache(maxsize=25)
            def func(values, args):
                nonlocal ncalls
                ncalls += 1
                swap_sign = args

                with set_values(all_params, values):
                    try:
                        loss_value, gradients = loss.value_gradients(params=other_params)
                    except tf.errors.InvalidArgumentError:
                        msg = (f"The evaluation of the errors of {param.name} failed due to too many NaNs"
                               " being produced in the loss and/or its gradients. This is most probably"
                               " caused by negative values returned from the PDF.")
                        raise FailEvalLossNaN(msg)

                    zeroed_loss = loss_value.numpy() - fmin

                    gradients = np.array(gradients)
                if swap_sign(param):
                    zeroed_loss = - zeroed_loss
                    gradients = - gradients
                    logging.info("Swapping sign in error calculation 'zfit_error'")

                elif zeroed_loss < - minimizer.tol:
                    set_values(all_params, values)  # set values to the new minimum
                    raise NewMinimum("A new is minimum found.")

                downward_shift = errordef * sigma ** 2
                shifted_loss = zeroed_loss - downward_shift

                return np.concatenate([[shifted_loss], gradients])

            to_return[param] = {}
            swap_sign = {
                "lower": lambda p: p > param_value,
                "upper": lambda p: p < param_value,
            }
            for d in ["lower", "upper"]:
                roots = optimize.root(fun=func,
                                      args=(swap_sign[d],),
                                      x0=initial_values[d],
                                      tol=rtol,
                                      options={
                                          # 'factor': 1.,
                                          # 'diag': 1 / param_scale,
                                      },
                                      method=method)
                to_return[param][d] = roots.x[all_params.index(param)] - param_value
        # print(f"errors found, time needed {time.time() - start}")

    except NewMinimum as e:
        from .. import settings
        if settings.get_verbosity() >= 5:
            print(e)
        minimizer = result.minimizer
        loss = result.loss
        new_found_fmin = loss.value()
        new_result = minimizer.minimize(loss=loss)
        if new_result.fmin >= new_found_fmin:
            raise RuntimeError("A new minimum was discovered but the minimizer was not able to find this on himself. "
                               "This behavior is currently an exception but will most likely change in the future.")
        to_return, new_result_ = compute_errors(result=new_result, params=params, sigma=sigma, rtol=rtol,
                                                method=method)
        if new_result_ is not None:
            new_result = new_result_
    print(f"Used {ncalls} calls.")
    return to_return, new_result


def numerical_pdf_jacobian(func, params):
    jacobian_func = numdifftools.Jacobian(func)
    jacobian = jacobian_func([param.value() for param in params]).T
    return jacobian


@z.function(wraps='autodiff')
def autodiff_pdf_jacobian(func, params):
    # with tf.GradientTape(persistent=False,
    #                      watch_accessed_variables=False) as tape:
    #     tape.watch(params)
    #     values = func()
    # jacobian = z.convert_to_tensor(tape.jacobian(values, params, experimental_use_pfor=False))
    # return jacobian

    columns = []

    for p in params:
        vector = np.zeros(len(params))
        vector[params.index(p)] = 1.
        with tf.autodiff.ForwardAccumulator(params, list(vector)) as acc:
            values = func()
        columns.append(acc.jvp(values))

    jacobian = z.convert_to_tensor(columns)

    return jacobian


def covariance_with_weights(method, result, params):
    model = result.loss.model
    data = result.loss.data

    Hinv_dict = method(result=result, params=params)  # inverse of the hessian matrix
    Hinv = dict_to_matrix(params, Hinv_dict)

    def func():
        values = []
        for m, d in zip(model, data):
            v = m.log_pdf(d)
            if d.weights is not None:
                v *= d.weights
            values.append(v)
        return tf.concat(values, axis=0)

    if settings.options['numerical_grad']:
        def wrapped_func(values):
            with set_values(params, values):
                return func()

        jacobian = numerical_pdf_jacobian(func=wrapped_func, params=params)
    else:
        jacobian = autodiff_pdf_jacobian(func=func, params=params).numpy()

    C = np.matmul(jacobian, jacobian.T)
    covariance = np.matmul(Hinv, np.matmul(C, Hinv))

    return matrix_to_dict(params, covariance)


def dict_to_matrix(params, matrix_dict):
    nparams = len(params)
    matrix = np.empty((nparams, nparams))

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

    for i in range(nparams):
        pi = params[i]
        for j in range(i, nparams):
            pj = params[j]
            matrix_dict[(pi, pj)] = matrix[i, j]
            if i != j:
                matrix_dict[(pj, pi)] = matrix[i, j]

    return matrix_dict


def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array

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
            """Iterates recursivelly."""
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator

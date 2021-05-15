#  Copyright (c) 2021 zfit
import logging
from functools import lru_cache, wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import numdifftools
import numpy as np
import scipy.stats
import tensorflow as tf
from scipy import optimize

import zfit.z.numpy as znp

from .. import settings, z
from ..core.interfaces import ZfitIndependentParameter
from ..core.parameter import assign_values
from ..param import set_values
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_args


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""
    pass


class FailEvalLossNaN(Exception):
    pass


@deprecated_args(None, "Use cl for confidence level instead.", 'sigma')
def compute_errors(result: "zfit.result.FitResult",
                   params: List[ZfitIndependentParameter],
                   cl: Optional[float] = None,
                   rtol: Optional[float] = 0.001,
                   method: Optional[str] = None,
                   covariance_method: Optional[Union[str, Callable]] = None,
                   sigma: float = 1,
                   ) -> Tuple[Dict[ZfitIndependentParameter, Dict[str, float]],
                              Union["zfit.result.FitResult", None]]:
    """Compute asymmetric errors of parameters by profiling the loss function in the fit result.

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
        rtol: relative tol between the computed and the exact roots
        method: type of solver, `method` argument of :py:func:`scipy.optimize.root`. Defaults to "hybr".
        covariance_method: The method to use to calculate the correlation matrix, will be forwarded directly
            to :py:meth:`FitResult.covariance`. Valid choices are
            by default {'minuit_hesse', 'hesse_np'} (or any other method defined in the result)
            or a Callable.
        sigma: Errors are calculated with respect to `sigma` std deviations.


    Returns:
        out:
            A `dict` containing as keys the parameter and as value a `dict` which
            contains two keys 'lower' and 'upper', holding the calculated errors.
            Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        out: a fit result is returned when a new minimum is found during the loss scan
    """
    method = "hybr" if method is None else method
    # method = "krylov" if method is None else method  # TODO: integration tests, better for large n params?
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
    from zfit import run
    old_values = run(result)

    covariance = result.covariance(method=covariance_method, as_dict=True)
    param_errors = {param: covariance[(param, param)] ** 0.5 for param in params}
    # param_scale = np.array(list(param_errors.values()))  # TODO: can be used for root finding initialization?

    ncalls = 0
    try:
        # start = time.time()
        to_return = {}
        for param in params:
            assign_values(all_params, result)

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
                    ap_value_init = ap_value + direction[d] * error_factor
                    initial_values[d].append(ap_value_init)

            # TODO: improvement, use jacobian?
            # @np_cache(maxsize=25)
            def func(values, args):
                nonlocal ncalls
                ncalls += 1
                swap_sign = args

                assign_values(all_params, values)
                try:
                    loss_value, gradient = loss.value_gradient(params=other_params)
                except tf.errors.InvalidArgumentError:
                    msg = (f"The evaluation of the errors of {param.name} failed due to too many NaNs"
                           " being produced in the loss and/or its gradient. This is most probably"
                           " caused by negative values returned from the PDF.")
                    raise FailEvalLossNaN(msg)

                zeroed_loss = loss_value.numpy() - fmin

                gradient = np.array(gradient)
                if swap_sign(param):  # mirror at x-axis to remove second zero
                    zeroed_loss = - zeroed_loss
                    gradient = - gradient
                    logging.info("Swapping sign in error calculation 'zfit_error'")

                elif zeroed_loss < - minimizer.tol:
                    assign_values(all_params, values)  # set values to the new minimum
                    raise NewMinimum("A new minimum is found.")

                downward_shift = errordef * sigma ** 2
                shifted_loss = zeroed_loss - downward_shift

                return np.concatenate([[shifted_loss], gradient])

            to_return[param] = {}
            swap_sign = {
                "lower": lambda p: p > param_value,
                "upper": lambda p: p < param_value,
            }
            for d in ["lower", "upper"]:
                roots = optimize.root(fun=func,
                                      args=(swap_sign[d],),
                                      x0=np.array(initial_values[d]),
                                      tol=rtol,
                                      options={
                                          'factor': 0.1,
                                          # 'diag': 1 / param_scale,  # scale factor for variables
                                          # 'diag': param_scale,
                                      },
                                      method=method)
                to_return[param][d] = roots.x[all_params.index(param)] - param_value
                # print(f"error {d}, time needed {time.time() - start2}")
        # print(f"errors found, time needed {time.time() - start}")
        assign_values(all_params, old_values)

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
    return to_return, new_result


def numerical_pdf_jacobian(func, params):
    jacobian_func = numdifftools.Jacobian(func)
    return jacobian_func([param.value() for param in params]).T


@z.function(wraps='autodiff')
def autodiff_pdf_jacobian(func, params):
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

    if settings.options['numerical_grad']:
        def wrapped_func(values):
            assign_values(params, values)
            return func()

        jacobian = numerical_pdf_jacobian(func=wrapped_func, params=params)
    else:
        jacobian = autodiff_pdf_jacobian(func=func, params=params).numpy()

    C = np.matmul(jacobian, jacobian.T)
    covariance = np.matmul(Hinv, np.matmul(C, Hinv))
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

#  Copyright (c) 2025 zfit

from __future__ import annotations

import logging
import typing
import warnings
from collections.abc import Callable
from enum import Enum
from functools import lru_cache, wraps

import jacobi
import numpy as np
import scipy.stats
import tensorflow as tf
from scipy import optimize

import zfit.z.numpy as znp
from zfit._interfaces import ZfitIndependentParameter

from .. import z
from ..core.parameter import assign_values
from ..util.container import convert_to_container
from ..util.exception import BreakingAPIChangeError

if typing.TYPE_CHECKING:
    import zfit


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""


class FailEvalLossNaN(Exception):
    pass


class RootFound(Exception):
    """Exception class for cases where a root is found, since SciPy root solvers don't really respect tol or xtol on
    initial evaluation."""


# make enum of WeightCorr
class WeightCorr(Enum):
    ASYMPTOTIC: str = "asymptotic"
    FALSE: bool = False
    SUMW2: str = "sumw2"  # sum of weights squared, not the sum of squares of weights


def compute_errors(
    result: zfit.result.FitResult,
    params: list[ZfitIndependentParameter],
    *,
    cl: float | None = None,
    rtol: float | None = None,
    method: str | None = None,
    covariance_method: str | Callable | None = None,
    sigma: float | None = None,
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
        sigma: Number of sigmas to calculate the error. Alternative to ``cl``.
        cl: Confidence Level of the parameter to be determined. Defaults to 68.3%. Alternative to ``sigma``.
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
        rtol = 0.003
    method = "hybr" if method is None else method
    if cl is None:
        if sigma is None:
            sigma = 1

    elif sigma is None:
        sigma = scipy.stats.chi2(1).ppf(cl) ** 0.5
    else:
        msg = "Cannot specify both sigma and cl."
        raise ValueError(msg)

    params = convert_to_container(params)
    new_result = None

    all_params = list(result.params.keys())
    loss = result.loss
    errordef = loss.errordef
    fmin = result.fminopt
    rtol *= errordef
    minimizer = result.minimizer

    covariance = result.covariance(method=covariance_method, as_dict=True)
    if covariance is None:
        covariance = result.covariance(method="hesse_np", as_dict=True)
    if covariance is None:
        msg = "Could not compute covariance matrix. Check if the minimum is valid."
        raise RuntimeError(msg)
    param_errors = {param: covariance[(param, param)] ** 0.5 for param in all_params}
    param_scale = np.array(list(param_errors.values()))

    ncalls = 0
    loss_min_tol = minimizer.tol * errordef * 2  # 2 is just to be tolerant
    try:
        to_return = {}
        for param in params:
            assign_values(all_params, result)

            logging.info(f"profiling the parameter {param}")  # noqa: G004
            param_error = param_errors[param]
            param_value = result.params[param]["value"]

            initial_values = {"lower": [], "upper": []}
            direction = {"lower": -1, "upper": 1}

            for ap in all_params:
                ap_value = result.params[ap]["value"]
                error_factor = covariance[(param, ap)] * (2 * errordef / param_error**2) ** 0.5
                for d in ["lower", "upper"]:
                    step = direction[d] * error_factor * sigma
                    for ntrial in range(50):  # noqa: B007
                        ap_value_init = ap_value + step
                        if ap_value_init < ap.lower or ap_value_init > ap.upper:
                            step *= 0.8
                        else:
                            break
                    else:
                        msg = (
                            f"Could not find a valid initial value for {ap} in {d} direction after {ntrial + 1} trials."
                            f" step tried: {step}. This should not happes, the error probably looks weird. Maybe plot"
                            f" the loss function for different parameter values and check if it looks reasonable."
                        )
                        raise RuntimeError(msg)

                    initial_values[d].append(ap_value_init)

            index_poi = all_params.index(param)  # remember the index
            _ = loss.value_gradient(params=all_params, full=False)  # to make sure the loss is compiled

            def make_optimized_loss_gradient_func(index_poi):
                @z.function(wraps="gradient", keepalive=True)
                def wrappedfunc(values, index):
                    assert isinstance(index, int)
                    assign_values(all_params, values)
                    loss_value, gradient = loss.value_gradient(params=all_params, full=False)
                    if isinstance(gradient, tuple | list):
                        gradient = znp.asarray(gradient)
                    gradient = znp.concatenate([gradient[:index_poi], gradient[index_poi + 1 :]])
                    return loss_value, gradient

                return wrappedfunc

            optimized_loss_gradient = make_optimized_loss_gradient_func(index_poi)

            # TODO: improvement, use jacobian?
            root = None
            ntol = 999  # if it's right in the beginning, we think it's fine

            # TODO: should we add a "robust" or similar option to not skip this?
            # or evaluate and then decide ,maybe use krylov as it doesn't do a lot of calls in the beginning, it
            # approximates the jacobian
            def make_func(index_poi, optimized_loss_gradient, param):  # bind these values
                def wrappedfunc(values, args):
                    nonlocal ncalls, root, ntol

                    ncalls += 1
                    swap_sign = args

                    try:
                        loss_value, gradient = optimized_loss_gradient(values, index_poi)
                    except tf.errors.InvalidArgumentError:
                        loss_value = znp.array(9999999999.0)
                        gradient = z.random.normal(stddev=0.1, shape=(len(all_params) - 1,))
                        # raise FailEvalLossNaN(msg)
                    zeroed_loss = loss_value - fmin

                    gradient = znp.array(gradient)

                    if swap_sign(param):  # mirror at x-axis to remove second zero
                        zeroed_loss = -zeroed_loss
                        gradient = -gradient
                        logging.info("Swapping sign in error calculation 'zfit_error'")

                    elif zeroed_loss < -loss_min_tol:
                        assign_values(all_params, values)  # set values to the new minimum
                        msg = "A new minimum is found."
                        raise NewMinimum(msg)

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

                return wrappedfunc

            func = make_func(index_poi, optimized_loss_gradient, param)

            to_return[param] = {}
            swap_sign = {
                "lower": lambda p, pval=param_value: p > pval,
                "upper": lambda p, pval=param_value: p < pval,
            }
            for d in ["lower", "upper"]:
                try:
                    root_result = optimize.root(
                        fun=func,
                        args=(swap_sign[d],),
                        x0=np.array(initial_values[d]),
                        tol=rtol * 0.1,  # we won't stop like this anyway
                        options={
                            "diag": 1 / param_scale,  # scale factor for variables
                        },
                        method=method,
                    )
                except RootFound:
                    assert root is not None, "Should be changed inside function."
                else:
                    warnings.warn(
                        f"The root finding did not converge below {rtol} but stopped by its own criteria.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    root = root_result.x[index_poi]
                to_return[param][d] = root - param_value

        assign_values(all_params, result)

    except NewMinimum:
        from .. import settings  # noqa: PLC0415

        if settings.get_verbosity() >= 5:
            pass
        minimizer = result.minimizer
        loss = result.loss
        new_found_fmin = loss.value(full=False)
        new_result = minimizer.minimize(loss=loss)
        if new_result.fminopt >= new_found_fmin + loss_min_tol:
            msg = (
                "A new minimum was discovered but the minimizer was not able to find this on himself. "
                "This behavior is currently an exception but will most likely change in the future."
            )
            raise RuntimeError(msg) from None
        to_return, new_result_ = compute_errors(result=new_result, params=params, cl=cl, rtol=rtol, method=method)
        if new_result_ is not None:
            new_result = new_result_
    return to_return, new_result


def numerical_pdf_jacobian(func, params):  # TODO: jit?
    """
    Args:
        func: The function for which the Jacobian will be computed.
        params: A dictionary of parameter values for the function.

    Returns:
        The numerical Jacobian of the given function with respect to the parameters.
    """
    params = list(params.values())
    return znp.asarray(jacobi.jacobi(func, params)[0].T)


@z.function(wraps="autodiff")
def autodiff_pdf_jacobian(func, params):
    """Computes the Jacobian matrix of a function using automatic differentiation.

    Args:
        func: A callable representing the function for which the Jacobian is to be calculated.
        params: A dictionary of parameters with their values that are passed to the function.
    """
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

    return znp.asarray(columns)


# TODO: refactor below, extract separate methods
def covariance_with_weights(hinv, result, params, *, weightcorr: WeightCorr = None):
    """Compute the covariance matrix of the parameters with weights.

    Args:
        hinv: The inverse of the Hessian matrix.
        result: The fit result.
        params: The parameters for which the covariance matrix is to be calculated.
        weightcorr: |@doc:result.hesse.weightcorr.method| Method to correct the estimation of the covariance matrix/hesse error
                   for a weighted likelihood. The following methods are available, for a comparison and
                   the derivation of the methods, see [langenbruch1]_:

                    - `False`: no correction, the covariance matrix is calculated as if the likelihood
                      was unweighted. This will generally underestimate the errors.
                    - `asymptotic`: the covariance matrix is corrected by the asymptotic formula
                      for the weighted likelihood. This is the default, yet computationally most
                      expensive method.
                    - `sumw2`: the covariance matrix is corrected by the effective sample size.
                      This is the fastest method but won't yield asymptotically correct results.

                   This is not (yet) guaranteed to fully work for binned fits and maybe under/over
                   represents errors.

                    .. [langenbruch1] Langenbruch, C. Parameter uncertainties in weighted unbinned maximum
                       likelihood fits
                       `Eur. Phys. J. C 82, 393 (2022). <https://doi.org/10.1140/epjc/s10052-022-10254-8>`_. |@docend:result.hesse.weightcorr.method|
    """
    from .. import run  # noqa: PLC0415

    if weightcorr == "sumw2":
        msg = "The 'sumw2' option has been renamed to 'sumw2'."
        raise BreakingAPIChangeError(msg)
    weightcorr = WeightCorr.ASYMPTOTIC if weightcorr is None else WeightCorr(weightcorr)

    run.assert_executing_eagerly()
    loss = result.loss
    model = loss.model
    data = loss.data
    yields = None
    if loss.is_extended:
        yields = []
        for m in model:
            yields.append(m.get_yield())
    constraints = None
    if loss.constraints:
        constraints = loss.constraints

    old_vals = np.asarray(params)

    Hinv_dict = hinv(result=result, params=params)  # inverse of the hessian matrix
    if not Hinv_dict:
        return {}
    Hinv = dict_to_matrix(params, Hinv_dict)
    dataweights = [
        d.weights if d.has_weights else znp.ones((d.nevents,))  # sum(ones_nevents ** 2) = nevents
        for d in data
    ]
    # extendedweights = [znp.reshape(znp.sum(w), (-1,)) for w in dataweights] if yields else []
    # extendedweights = [znp.ones((len(yields),))] if yields else []
    constrweights = [znp.ones((len(constraints),))] if constraints else []
    allweights = znp.concatenate(dataweights + constrweights, axis=0)
    allweights2 = allweights**2
    sumw2 = znp.sum(allweights2)
    sumw = znp.sum(allweights)
    if weightcorr == WeightCorr.ASYMPTOTIC:
        corrfactor = 1.0
    elif weightcorr == WeightCorr.SUMW2:
        corrfactor = sumw2 / sumw
        del allweights
    else:
        msg = f"Unknown method {weightcorr}, has to be one of {WeightCorr}"
        raise ValueError(msg)

    @z.function(wraps="weightcorrfunc")
    def func():
        values = []

        for i, (m, d) in enumerate(zip(model, data, strict=True)):
            v = m.log_pdf(d)
            # we calculate the unweighted likelihood, correct?
            # weights = d.weights
            # print(f"weights: {weights}, corrfactor: {corrfactor}")
            # if weights is not None:
            #     print(f"weights: {weights}, corrfactor: {corrfactor}")
            #     v *= weights

            if yields is not None:
                yi = yields[i]
                # shouldn't the normal yield be enough? Not quite sure why not
                # Probably because it's a sum of the weights
                # nevents_collected = d.samplesize
                # extendedterm = tf.nn.log_poisson_loss(nevents_collected, znp.log(yi), compute_full_loss=True)
                # values.append(term_new)
                extendedterm = znp.log(yi)
                v += extendedterm
                # values.append(extendedterm[..., None])  # make it an array like the others
            values.append(v)
        if constraints is not None:
            for constraint in constraints:
                values.append(znp.reshape(constraint.value(), (-1,)))

        return znp.concatenate(values, axis=0)

    params_dict = {p.name: p for p in params}
    if weightcorr == WeightCorr.ASYMPTOTIC:
        if run.get_autograd_mode():
            try:
                jacobian = autodiff_pdf_jacobian(func=func, params=params_dict)
            except ValueError as error:
                msg = (
                    "The autodiff could not be performed for the jacobian matrix. This can have a natural cause (see error above)"
                    " or be a bug in the autodiff. In the latter case, you can try to use the numerical jacobian instead using:\n"
                    "with zfit.run.set_autograd_mode(False):\n"
                    "    # calculate here, i.e. result.errors()"
                )
                raise ValueError(msg) from error
        else:

            def wrapped_func(values):
                assign_values(list(params_dict.values()), values)
                return np.array(func())

            jacobian = numerical_pdf_jacobian(func=wrapped_func, params=params_dict)

        C = znp.matmul(jacobian * allweights2, znp.transpose(jacobian))

        covariance = np.asarray(znp.matmul(Hinv, znp.matmul(C, Hinv)))
    elif weightcorr == WeightCorr.SUMW2:
        # Not quite correct, technically. The weights should be squared in the
        # NLL calculation
        covariance = Hinv * corrfactor
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

#  Copyright (c) 2020 zfit

import numpy as np
from scipy import optimize

from ..param import set_values
from ..util.container import convert_to_container


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""
    pass


def pll(minimizer, loss, params, values) -> 'FitResult':
    """Compute minimum profile likelihood for given parameters and values."""
    params = convert_to_container(params)
    values = convert_to_container(values)

    verbosity = minimizer.verbosity
    minimizer.verbosity = 0

    with set_values(params, values):
        for param in params:
            param.floating = False

        minimum = minimizer.minimize(loss=loss)

    for param in params:
        param.floating = True
    minimizer.verbosity = verbosity

    return minimum


def get_crossing_value(result, params, direction, sigma, rootf, rtol):
    """Find the crossing point between the profiled loss function, for given parameters, and the value of
    `errordef` for a given direction (positive / negative).
    `errordef` = 1 for a chisquare fit, = 0.5 for a likelihood fit.
    """

    all_params = list(result.params.keys())
    loss = result.loss
    errordef = loss.errordef
    fmin = result.fmin
    minimizer = result.minimizer.copy()
    minimizer.tolerance = minimizer.tolerance * 0.5
    rtol *= errordef

    set_values(all_params, result)

    covariance = result.covariance(as_dict=True)

    step_size_dict = {ap: ap.step_size for ap in all_params}
    for ap in all_params:
        ap.step_size = covariance[(ap, ap)] ** 0.5

    to_return = {}
    for param in params:
        param_error = result.hesse(params=param)[param]["error"]
        param_value = result.params[param]["value"]
        exp_root = param_value + sigma * direction * param_error  # expected root

        for ap in all_params:
            if ap == param:
                continue

            # shift parameters, other than param, using covariance matrix
            ap_value = result.params[ap]["value"]
            ap_value += sigma * direction * covariance[(param, ap)] * (2 * errordef / param_error ** 2) ** 0.5
            ap.set_value(ap_value)

        cache = {}  # TODO: round for better floating point comparison?

        def shifted_pll(v):
            """
            Computes the pll, with the minimum substracted and shifted by minus the `errordef`, for a
            given parameter.
            `errordef` = 1 for a chisquare fit, = 0.5 for a likelihood fit.

            The function raises a `NewMinimum` exception if the value of the shifted `pll` is less than
            `- errordef`.
            """
            if v not in cache:
                pll_result = pll(minimizer, loss, param, v)
                cache[v] = pll_result.fmin - fmin - errordef
                if cache[v] < -errordef - minimizer.tolerance:
                    set_values(pll_result.params.keys(), pll_result)  # set values to the new minimum
                    raise NewMinimum("A new is minimum found.")

            return cache[v]

        exp_shifted_pll = shifted_pll(exp_root)

        if np.allclose(0., exp_shifted_pll, atol=0.0005):
            root = exp_root

        else:
            def linear_interp(y):
                """
                Linear interpolation between the minimum of the `shifted_pll` curve and its expected root,
                assuming it is a parabolic curve.
                """
                slope = (exp_root - param_value) / (exp_shifted_pll + errordef)
                return param_value + (y + errordef) * slope

            bound_interp = linear_interp(0)

            if exp_shifted_pll > 0.:
                lower_bound = exp_root
                upper_bound = bound_interp
            else:
                lower_bound = bound_interp
                upper_bound = exp_root

            if direction == 1:
                lower_bound, upper_bound = upper_bound, lower_bound

            # Check if the `shifted_pll` function has the same sign at the lower and upper bounds.
            # If they have the same sign, the window given to the root finding algorithm is increased.

            nsigma = 1.5
            while np.sign(shifted_pll(lower_bound)) == np.sign(shifted_pll(upper_bound)):
                if direction == -1:
                    if np.sign(shifted_pll(lower_bound)) == -1:
                        lower_bound = param_value - nsigma * param_error
                    else:
                        upper_bound = param_value
                else:
                    if np.sign(shifted_pll(lower_bound)) == -1:
                        upper_bound = param_value + nsigma * param_error
                    else:
                        lower_bound = param_value

                nsigma += 0.5

            root, results = rootf(f=shifted_pll, a=lower_bound, b=upper_bound, rtol=rtol,
                                  full_output=True)

        to_return[param] = root

    for ap in all_params:
        ap.step_size = step_size_dict[ap]

    return to_return


def _rootf(**kwargs):
    return optimize.toms748(k=1, **kwargs)


# def _rootf(**kwargs):
#     return optimize.brentq(**kwargs)

def compute_errors_old(result, params, sigma=1, rootf=_rootf, rtol=0.01):
    """
    Computes asymmetric errors of parameters by profiling the loss function in the fit result.

    Args:
        result (`FitResult`): fit result
        params (list(:py:class:`~zfit.Parameter`)): The parameters to calculate the
            errors error. If None, use all parameters.
        sigma (float): Errors are calculated with respect to `sigma` std deviations.
        rootf (callable): function used to find the roots of the loss function
        rtol (float, default=0.01): relative tolerance between the computed and the exact roots

    Returns:
        `OrderedDict`: A `OrderedDict` containing as keys the parameter and as value a `dict` which
            contains two keys 'lower' and 'upper', holding the calculated errors.
            Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        `FitResult` or `None`: a fit result is returned when a new minimum is found during the loss scan
    """

    params = convert_to_container(params)
    new_result = None

    try:

        upper_values = get_crossing_value(result=result, params=params, direction=1, sigma=sigma,
                                          rootf=rootf, rtol=rtol)

        lower_values = get_crossing_value(result=result, params=params, direction=-1, sigma=sigma,
                                          rootf=rootf, rtol=rtol)

        to_return = {}
        for param in params:
            fitted_value = result.params[param]["value"]
            to_return[param] = {"lower": lower_values[param] - fitted_value,
                                "upper": upper_values[param] - fitted_value}

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
        to_return, new_result_ = compute_errors_old(result=new_result, params=params, sigma=sigma,
                                                rootf=rootf, rtol=rtol)
        if new_result_ is not None:
            new_result = new_result_

    return to_return, new_result


def compute_errors(result, params, sigma=1, rtol=0.001, method="hybr"):
    """
    Computes asymmetric errors of parameters by profiling the loss function in the fit result.

    Args:
        result (`FitResult`): fit result
        params (list(:py:class:`~zfit.Parameter`)): The parameters to calculate the
            errors error. If None, use all parameters.
        sigma (float): Errors are calculated with respect to `sigma` std deviations.
        rtol (float, default=0.01): relative tolerance between the computed and the exact roots
        method (string, defautl='hybr'): type of solver, `method` argument of `scipy.optimize.root`

    Returns:
        `OrderedDict`: A `OrderedDict` containing as keys the parameter and as value a `dict` which
            contains two keys 'lower' and 'upper', holding the calculated errors.
            Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        `FitResult` or `None`: a fit result is returned when a new minimum is found during the loss scan
    """

    params = convert_to_container(params)
    new_result = None

    all_params = list(result.params.keys())
    loss = result.loss
    errordef = loss.errordef
    fmin = result.fmin
    rtol *= errordef
    minimizer = result.minimizer

    covariance = result.covariance(as_dict=True)
    set_values(all_params, result)

    try:

        to_return = {}
        for param in params:
            param_error = result.hesse(params=param)[param]["error"]
            param_value = result.params[param]["value"]
            other_params = [p for p in all_params if p != param]

            initial_values = {"lower": [], "upper": []}
            direction = {"lower": -sigma, "upper": sigma}

            for ap in all_params:
                for d in ["lower", "upper"]:
                    # shift parameters, other than param, using covariance matrix
                    ap_value = result.params[ap]["value"]
                    ap_value += direction[d] * covariance[(param, ap)] * (2 * errordef / param_error ** 2) ** 0.5
                    initial_values[d].append(ap_value)

            def func(values):

                with set_values(all_params, values):
                    loss_value, gradients = loss.value_gradients(params=other_params)
                    shifted_loss = loss_value.numpy() - fmin - errordef
                    gradient = np.array(gradients)

                if shifted_loss < -errordef - minimizer.tolerance:
                    set_values(all_params, values)  # set values to the new minimum
                    raise NewMinimum("A new is minimum found.")

                return np.concatenate([[shifted_loss], gradient])

            to_return[param] = {}
            for d in ["lower", "upper"]:
                roots = optimize.root(fun=func, x0=initial_values[d], tol=rtol, method=method)
                to_return[param][d] = roots.x[all_params.index(param)] - param_value

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

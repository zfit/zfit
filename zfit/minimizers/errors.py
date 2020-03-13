# import numpy as np
from scipy import optimize
from ..param import set_values
from ..util.container import convert_to_container


def pll(minimizer, loss, params, values) -> float:
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

    return minimum.fmin


def set_params_to_result(params, result):
    for param in params:
        param.set_value(result.params[param]["value"])


def get_crossing_value(result, params, direction, sigma, rootf):

    all_params = list(result.params.keys())
    loss = result.loss
    up = loss.errordef
    fmin = result.fmin
    minimizer = result.minimizer.copy()
    minimizer.minimizer_options["strategy"] = max(0, minimizer.minimizer_options["strategy"] - 1)
    minimizer.tolerance = 0.01
    rtol = 0.0005

    set_params_to_result(all_params, result)

    covariance = result.covariance(as_dict=True)
    sigma = sigma * direction

    to_return = {}
    for param in params:
        cache = {}
        param_error = result.hesse(params=param)[param]["error"]
        param_value = result.params[param]["value"]
        param_value_sigma = param_value + sigma * param_error

        for ap in all_params:
            if ap == param:
                continue

            ap_value = result.params[ap]["value"]
            ap_error = covariance[(ap, ap)] ** 0.5
            ap_value += sigma ** 2 * covariance[(param, ap)] / ap_error
            ap.set_value(ap_value)

        def shifted_pll(v):
            if v not in cache:
                cache[v] = pll(minimizer, loss, param, v) - fmin - up

            return cache[v]

        exp_shifted_pll = shifted_pll(param_value_sigma)

        def linear_interp(y):
            return param_value + (y + up) * (param_value_sigma - param_value) / (exp_shifted_pll + up)
        bound_interp = linear_interp(0)

        if exp_shifted_pll > 0.:
            lower_bound = param_value_sigma
            upper_bound = bound_interp
        else:
            lower_bound = bound_interp
            upper_bound = param_value_sigma

        if direction == 1:
            lower_bound, upper_bound = upper_bound, lower_bound

        root, results = rootf(f=shifted_pll, a=lower_bound, b=upper_bound, rtol=rtol, full_output=True)

        to_return[param] = root

    return to_return


# def _rootf(**kwargs):
#     return optimize.toms748(k=2, **kwargs)

def _rootf(**kwargs):
    return optimize.brentq(**kwargs)


def compute_errors(result, params, sigma=1, rootf=_rootf):

    params = convert_to_container(params)

    upper_values = get_crossing_value(result=result, params=params, direction=1, sigma=sigma,
                                      rootf=rootf)

    lower_values = get_crossing_value(result=result, params=params, direction=-1, sigma=sigma,
                                      rootf=rootf)

    to_return = {}
    for param in params:
        fitted_value = result.params[param]["value"]
        to_return[param] = {"lower": lower_values[param] - fitted_value,
                            "upper": upper_values[param] - fitted_value,
                            }

    return to_return

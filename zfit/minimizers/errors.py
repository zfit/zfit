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


def get_crossing_value(result, params, direction, sigma):

    all_params = list(result.params.keys())
    loss = result.loss
    up = loss.errordef
    fmin = result.fmin
    minimizer = result.minimizer
    sigma = sigma * direction

    to_return = {}
    for param in params:
        set_params_to_result(all_params, result)
        param_error = result.hesse(params=param)[param]["error"]
        param_value = result.params[param]["value"]

        if direction == -1:
            lower_bound = param_value + 2 * param_error * sigma
            upper_bound = param_value
        else:
            lower_bound = param_value
            upper_bound = param_value + 2 * param_error * sigma

        root, results = optimize.toms748(f=lambda v: pll(minimizer, loss, param, v) - fmin - up,
                                         a=lower_bound, b=upper_bound, full_output=True, k=2)
        to_return[param] = root

    return to_return


def compute_errors(result, params, sigma=1):

    params = convert_to_container(params)

    upper_values = get_crossing_value(result=result, params=params, direction=1, sigma=sigma)

    lower_values = get_crossing_value(result=result, params=params, direction=-1, sigma=sigma)

    to_return = {}
    for param in params:
        fitted_value = result.params[param]["value"]
        to_return[param] = {"lower": lower_values[param] - fitted_value,
                            "upper": upper_values[param] - fitted_value,
                            }

    return to_return

#  Copyright (c) 2020 zfit

import numpy as np
import tensorflow as tf
from scipy import optimize
import itertools
import numdifftools

from .. import z, settings
from ..param import set_values
from ..util.container import convert_to_container


class NewMinimum(Exception):
    """Exception class for cases where a new minimum is found."""
    pass


class FailEvalLossNaN(Exception):
    pass


def compute_errors(result, params, sigma=1, rtol=0.001, method="hybr", covariance_method=None):
    """
    Computes asymmetric errors of parameters by profiling the loss function in the fit result.

    Args:
        result: fit result
        params: The parameters to calculate the
            errors error. If None, use all parameters.
        sigma: Errors are calculated with respect to `sigma` std deviations.
        rtol: relative tolerance between the computed and the exact roots
        method: type of solver, `method` argument of `scipy.optimize.root_
        covariance_method: The method to use to calculate the correlation matrix. Valid choices are
            {'minuit_hesse', 'hesse_np'} or a Callable.

    Returns:
        A `OrderedDict` containing as keys the parameter and as value a `dict` which
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

    covariance = result.covariance(method=covariance_method, as_dict=True)
    set_values(all_params, result)

    try:

        to_return = {}
        for param in params:
            param_error = covariance[(param, param)]**0.5
            param_value = result.params[param]["value"]
            other_params = [p for p in all_params if p != param]

            initial_values = {"lower": [], "upper": []}
            direction = {"lower": -sigma, "upper": sigma}

            for ap in all_params:
                for d in ["lower", "upper"]:
                    ap_value = result.params[ap]["value"]
                    ap_value += direction[d] * covariance[(param, ap)] * (2 * errordef / param_error ** 2) ** 0.5
                    initial_values[d].append(ap_value)

            def func(values):

                with set_values(all_params, values):
                    try:
                        loss_value, gradients = loss.value_gradients(params=other_params)
                    except tf.errors.InvalidArgumentError:
                        msg = (f"The evaluation of the errors of {param.name} failed due to too many NaNs"
                               " being produced in the loss and/or its gradients. This is most probably"
                               " caused by negative values returned from the PDF.")
                        raise FailEvalLossNaN(msg)

                    shifted_loss = loss_value.numpy() - fmin - errordef * sigma**2
                    gradients = np.array(gradients)

                if shifted_loss < -errordef - minimizer.tolerance:
                    set_values(all_params, values)  # set values to the new minimum
                    raise NewMinimum("A new is minimum found.")

                return np.concatenate([[shifted_loss], gradients])

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


def numerical_pdf_jacobian(func, params):
    jacobian_func = numdifftools.Jacobian(func)
    jacobian = jacobian_func([param.value() for param in params]).T
    return jacobian


@z.function
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

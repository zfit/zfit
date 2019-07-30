#  Copyright (c) 2019 zfit

from collections import OrderedDict, defaultdict
from typing import Dict, Union, Callable, Optional
import warnings

import tensorflow as tf
import numpy as np

import zfit
from zfit.util.exception import WeightsNotImplementedError, DueToLazynessNotImplementedError
from zfit.util.execution import SessionHolderMixin
from .interface import ZfitMinimizer, ZfitResult
from ..util.ztyping import ParamsTypeOpt
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..util.temporary import TemporarilySet
from ..util.container import convert_to_container


def _hesse_minuit(result: "FitResult", params, sigma=1.0):
    if sigma != 1.0:
        raise ValueError("sigma other then 1 is not valid for minuit hesse.")
    fitresult = result

    # check if no weights in data
    if any([data.weights is not None for data in result.loss.data]):
        raise WeightsNotImplementedError("Weights are not supported with minuit hesse.")

    minimizer = fitresult.minimizer
    from zfit.minimizers.minimizer_minuit import Minuit
    if not isinstance(minimizer, Minuit):
        raise TypeError("Cannot perform hesse error calculation 'minuit' with a different minimizer then"
                        "`Minuit`.")
    params_name = OrderedDict((param.name, param) for param in params)
    result_hesse = minimizer._minuit_minimizer.hesse()
    result_hesse = OrderedDict((res['name'], res) for res in result_hesse)

    result = OrderedDict((params_name[p_name], {'error': res['error']})
                         for p_name, res in result_hesse.items() if p_name in params_name)
    return result


def _minos_minuit(result, params, sigma=1.0):
    fitresult = result
    minimizer = fitresult.minimizer
    from zfit.minimizers.minimizer_minuit import Minuit
    if not isinstance(minimizer, Minuit):
        raise TypeError("Cannot perform error calculation 'minos_minuit' with a different minimizer then"
                        "`Minuit`.")
    result = [minimizer._minuit_minimizer.minos(var=p.name, sigma=sigma)
              for p in params][-1]  # returns every var
    result = OrderedDict((p, result[p.name]) for p in params)
    return result


class FitResult(SessionHolderMixin, ZfitResult):
    _default_hesse = 'minuit_hesse'
    _hesse_methods = {'minuit_hesse': _hesse_minuit}
    _default_error = 'minuit_minos'
    _error_methods = {"minuit_minos": _minos_minuit}

    def __init__(self, params: Dict[ZfitParameter, float], edm: float, fmin: float, status: int, converged: bool,
                 info: dict, loss: ZfitLoss, minimizer: "ZfitMinimizer"):
        """Create a `FitResult` from a minimization. Store parameter values, minimization infos and calculate errors.

        Any errors calculated are saved under `self.params` dictionary with {parameter: {error_name1: {'low': value
        'high': value or similar}}

        Args:
            params (OrderedDict[:py:class:`~zfit.Parameter`, float]): Result of the fit where each
            :py:class:`~zfit.Parameter` key has the value
                from the minimum found by the minimizer.
            edm (Union[int, float]): The estimated distance to minimum, estimated by the minimizer (if available)
            fmin (Union[numpy.float64, float]): The minimum of the function found by the minimizer
            status (int): A status code (if available)
            converged (bool): Whether the fit has successfully converged or not.
            info (Dict): Additional information (if available) like *number of function calls* and the
                original minimizer return message.
            loss (Union[ZfitLoss]): The loss function that was minimized. Contains also the pdf, data etc.
            minimizer (ZfitMinimizer): Minimizer that was used to obtain this `FitResult` and will be used to
                calculate certain errors. If the minimizer is state-based (like "iminuit"), then this is a copy
                and the state of other `FitResults` or of the *actual* minimizer that performed the minimization
                won't be altered.
        """
        super().__init__()
        self._status = status
        self._converged = converged
        self._params = self._input_convert_params(params)
        self._edm = edm
        self._fmin = fmin
        self._info = info
        self._loss = loss
        self._minimizer = minimizer
        # self.param_error = OrderedDict((p, {}) for p in params)
        # self.param_hesse = OrderedDict((p, {}) for p in params)

    def _input_convert_params(self, params):
        params = OrderedDict((p, {'value': v}) for p, v in params.items())
        return params

    def _get_uncached_params(self, params, method_name):
        params_uncached = [p for p in params if self.params[p].get(method_name) is None]
        return params_uncached

    @property
    def params(self):
        return self._params

    @property
    def edm(self):
        """The estimated distance to the minimum.

        Returns:
            numeric
        """
        edm = self._edm
        return edm

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def loss(self):
        # TODO(Mayou36): this is currently a reference, should be a copy of the loss?
        return self._loss

    @property
    def fmin(self):
        """Function value at the minimum.

        Returns:
            numeric
        """
        fmin = self._fmin
        return fmin

    @property
    def status(self):
        status = self._status
        return status

    @property
    def info(self):
        return self._info

    @property
    def converged(self):
        return self._converged

    def _input_check_params(self, params):
        if params is not None:
            params = convert_to_container(params)
        else:
            params = list(self.params.keys())
        return params

    def hesse(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = 'minuit_hesse',
              error_name: Optional[str] = None) -> OrderedDict:
        """Calculate for `params` the symmetric error using the Hessian matrix.

        Args:
            params (list(:py:class:`~zfit.Parameter`)): The parameters  to calculate the
                Hessian symmetric error. If None, use all parameters.
            method (str): the method to calculate the hessian. Can be {'minuit'} or a callable.
            error_name (str): The name for the error in the dictionary.

        Returns:
            OrderedDict: Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'error': sym_error}.

                So given param_a (from zfit.Parameter(.))
                `error_a = result.hesse(params=param_a)[param_a]['error']`
                error_a is the hessian error.
        """
        if error_name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            error_name = method
        params = self._input_check_params(params)
        uncached_params = self._get_uncached_params(params=params, method_name=error_name)
        if uncached_params:
            error_dict = self._hesse(params=uncached_params, method=method)
            self._cache_errors(error_name=error_name, errors=error_dict)
        all_errors = OrderedDict((p, self.params[p][error_name]) for p in params)
        return all_errors

    def _cache_errors(self, error_name, errors):
        for param, errors in errors.items():
            self.params[param][error_name] = errors

    def _hesse(self, params, method):
        if not callable(method):
            try:
                method = self._hesse_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))
        return method(result=self, params=params)

    def error(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = 'minuit_minos', error_name: str = None,
              sigma: float = 1.) -> OrderedDict:

        """Calculate and set for `params` the asymmetric error using the set error method.

            Args:
                params (list(:py:class:`~zfit.Parameter` or str)): The parameters or their names to calculate the
                     errors. If `params` is `None`, use all *floating* parameters.
                method (str or Callable): The method to use to calculate the errors. Valid choices are
                    {'minuit_minos'} or a Callable.
                sigma (float): Errors are calculated with respect to `sigma` std deviations. The definition
                    of 1 sigma depends on the loss function and is defined there.

                    For example, the negative log-likelihood (without the factor of 2) has a correspondents
                    of :math:`\Delta` NLL of 1 corresponds to 1 std deviation.
                error_name (str): The name for the error in the dictionary.


            Returns:
                `OrderedDict`: A `OrderedDict` containing as keys the parameter names and as value a `dict` which
                    contains (next to probably more things) two keys 'lower' and 'upper',
                    holding the calculated errors.
                    Example: result['par1']['upper'] -> the asymmetric upper error of 'par1'
        """
        if error_name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            error_name = method
        params = self._input_check_params(params)
        uncached_params = self._get_uncached_params(params=params, method_name=error_name)

        if uncached_params:
            error_dict = self._error(params=uncached_params, method=method, sigma=sigma)
            self._cache_errors(error_name=error_name, errors=error_dict)
        all_errors = OrderedDict((p, self.params[p][error_name]) for p in params)
        return all_errors

    def _error(self, params, method, sigma):
        if not callable(method):
            try:
                method = self._error_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))
        return method(result=self, params=params, sigma=sigma)

    def covariance(self, params: ParamsTypeOpt = None, as_dict: bool = False):
        """Calculate the covariance matrix for `params`.

            Args:
                params (list(:py:class:`~zfit.Parameter`)): The parameters to calculate
                    the covariance matrix. If `params` is `None`, use all *floating* parameters.
                as_dict (bool): Default `False`. If `True` then returns a dictionnary.

            Returns:
                2D `numpy.array` of shape (N, N);
                `dict`(param1, param2) -> covariance if `as_dict == True`.
        """
        params = self._input_check_params(params)

        try:
            covariance_dict = self.minimizer._minuit_minimizer.covariance
        except AttributeError:
            raise DueToLazynessNotImplementedError("Currently, only covariance from minuit is available. "
                                                   "Use `Minuit` or open an issue on GitHub.")

        cov = {}
        for p1 in params:
            for p2 in params:
                key = (p1, p2)
                cov[key] = covariance_dict[tuple(k.name for k in key)]
        covariance_dict = cov

        if as_dict:
            return covariance_dict
        else:
            return dict_to_matrix(params, covariance_dict)


def dict_to_matrix(params, matrix_dict):

    nparams = len(params)
    matrix = np.empty((nparams, nparams))

    for i in range(nparams):
        pi = params[i]
        for j in range(i, nparams):
            pj = params[j]
            key = (pi, pj)
            matrix[i, j] = matrix_dict[key]
            if i != j:
                matrix[j, i] = matrix_dict[key]

    return matrix

# def set_error_method(self, method):
#     if isinstance(method, str):
#         try:
#             method = self._error_methods[method]
#         except AttributeError:
#             raise AttributeError("The error method '{}' is not registered with the minimizer.".format(method))
#     elif callable(method):
#         self._current_error_method = method
#     else:
#         raise ValueError("Method {} is neither a valid method name nor a callable function.".format(method))
#
# def set_error_options(self, replace: bool = False, **options):
#     """Specify the options for the `error` calculation.
#
#     Args:
#         replace (bool): If True, replace the current options. If False, only update
#             (add/overwrite existing).
#         **options (keyword arguments): The keyword arguments that will be given to `error`.
#     """
#         self._current_error_options = {}
#     self._current_error_options.update(options)

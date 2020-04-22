#  Copyright (c) 2020 zfit

import itertools
import warnings
from collections import OrderedDict
from typing import Dict, Union, Callable, Optional, Tuple

import colored
import numpy as np
import tableformatter as tafo
from colorama import Style
from ordered_set import OrderedSet

from .interface import ZfitMinimizer, ZfitResult
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..settings import run
from ..util.container import convert_to_container
from ..util.exception import WeightsNotImplementedError
from ..util.ztyping import ParamsTypeOpt

from .errors import compute_errors


def _minos_minuit(result, params, sigma=1.0):
    fitresult = result
    minimizer = fitresult.minimizer
    from zfit.minimizers.minimizer_minuit import Minuit

    if not isinstance(minimizer, Minuit):
        raise TypeError("Cannot perform error calculation 'minos_minuit' with a different minimizer than"
                        "`Minuit`.")

    result = [minimizer._minuit_minimizer.minos(var=p.name, sigma=sigma)
              for p in params][-1]  # returns every var
    result = OrderedDict((p, result[p.name]) for p in params)
    new_result = None
    return result, new_result


def _covariance_minuit(result, params):
    # check if no weights in data
    if any([data.weights is not None for data in result.loss.data]):
        raise WeightsNotImplementedError("Weights are not supported with minuit hesse.")

    fitresult = result
    minimizer = fitresult.minimizer

    from zfit.minimizers.minimizer_minuit import Minuit

    if not isinstance(minimizer, Minuit):
        raise TypeError("Cannot compute the covariance matrix with 'covariance_minuit' with a different"
                        " minimizer than `Minuit`.")

    covariance = result.minimizer._minuit_minimizer.covariance

    covariance_dict = {}
    for p1 in params:
        for p2 in params:
            key = (p1, p2)
            covariance_dict[key] = covariance[tuple(k.name for k in key)]

    return covariance_dict


def _covariance_np(result, params):
    # check if no weights in data
    if any([data.weights is not None for data in result.loss.data]):
        raise WeightsNotImplementedError("Weights are not supported with hesse numpy.")

    # TODO: maybe activate again? currently fails due to numerical problems
    # numgrad_was_none = settings.options.numerical_grad is None
    # if numgrad_was_none:
    #     settings.options.numerical_grad = True
    covariance = np.linalg.inv(result.loss.value_gradients_hessian(params)[2])
    # if numgrad_was_none:
    #     settings.options.numerical_grad = None

    return matrix_to_dict(params, covariance)


class FitResult(ZfitResult):
    _default_hesse = "hesse_np"
    _hesse_methods = {"minuit_hesse": _covariance_minuit, "hesse_np": _covariance_np}
    _default_error = "zfit_error"
    _error_methods = {"minuit_minos": _minos_minuit, "zfit_error": compute_errors}

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
        self._valid = True
        # self.param_error = OrderedDict((p, {}) for p in params)
        # self.param_hesse = OrderedDict((p, {}) for p in params)

    def _input_convert_params(self, params):
        params = ParamHolder((p, {"value": v}) for p, v in params.items())
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
    def loss(self) -> ZfitLoss:
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

    @property
    def valid(self):
        return self._valid

    def _input_check_params(self, params):
        if params is not None:
            params = convert_to_container(params)
        else:
            params = list(self.params.keys())
        return params

    def hesse(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None,
              error_name: Optional[str] = None) -> OrderedDict:
        """Calculate for `params` the symmetric error using the Hessian/covariance matrix.

        Args:
            params (list(:py:class:`~zfit.Parameter`)): The parameters  to calculate the
                Hessian symmetric error. If None, use all parameters.
            method (str): the method to calculate the covariance matrix. Can be {'minuit_hesse', 'hesse_np'} or a callable.
            error_name (str): The name for the error in the dictionary.

        Returns:
            OrderedDict: Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'error': sym_error}.

                So given param_a (from zfit.Parameter(.))
                `error_a = result.hesse(params=param_a)[param_a]['error']`
                error_a is the hessian error.
        """
        if method is None:
            # LEGACY START
            method = self._default_hesse
            from zfit.minimizers.minimizer_minuit import Minuit

            if isinstance(self.minimizer, Minuit):
                method = "minuit_hesse"
            # LEGACY END
        if error_name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            error_name = method

        all_params = list(self.params.keys())
        uncached_params = self._get_uncached_params(params=all_params, method_name=error_name)

        if uncached_params:
            error_dict = self._hesse(params=uncached_params, method=method)
            self._cache_errors(error_name=error_name, errors=error_dict)

        params = self._input_check_params(params)
        all_errors = OrderedDict((p, self.params[p][error_name]) for p in params)
        return all_errors

    def _cache_errors(self, error_name, errors):
        for param, errors in errors.items():
            self.params[param][error_name] = errors

    def _hesse(self, params, method):
        covariance_dict = self.covariance(params, method, as_dict=True)
        return OrderedDict((p, {"error": covariance_dict[(p, p)] ** 0.5}) for p in params)

    def error(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, error_name: str = None,
              sigma: float = 1.0) -> OrderedDict:
        r"""DEPRECATED! Use 'errors' instead
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
        warnings.warn("`error` is depreceated, use `errors` instead. This will return not only the errors but also "
                      "(a possible) new FitResult if a minimum was found. So change"
                      "errors = result.error()"
                      "to"
                      "errors, new_res = result.errors()", DeprecationWarning)
        return self.errors(params=params, method=method, error_name=error_name, sigma=sigma)[0]

    def errors(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, error_name: str = None,
               sigma: float = 1.0) -> Tuple[OrderedDict, Union[None, 'FitResult']]:
        r"""Calculate and set for `params` the asymmetric error using the set error method.

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
                `OrderedDict`: A `OrderedDict` containing as keys the parameter and as value a `dict` which
                    contains (next to probably more things) two keys 'lower' and 'upper',
                    holding the calculated errors.
                    Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        """
        if method is None:
            # TODO: legacy, remove 0.6
            from zfit.minimize import Minuit
            if isinstance(self.minimizer, Minuit):
                method = 'minuit_minos'
                warnings.warn("'minuit_minos' will be changed as the default errors method to a custom implementation"
                              "with the same functionality 'zfit_error'. If you want to make sure that 'minuit_minos' will be used "
                              "in the future, add it explicitly as in `errors(method='minuit_minos')`", FutureWarning)
            else:
                method = self._default_error
        if error_name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            error_name = method

        params = self._input_check_params(params)
        uncached_params = self._get_uncached_params(params=params, method_name=error_name)

        new_result = None

        if uncached_params:
            error_dict, new_result = self._error(params=uncached_params, method=method, sigma=sigma)
            if new_result is None:
                self._cache_errors(error_name=error_name, errors=error_dict)
            else:
                msg = "Invalid, a new minimum was found."
                self._cache_errors(error_name=error_name, errors={p: msg for p in params})
                self._valid = False
                new_result._cache_errors(error_name=error_name, errors=error_dict)
        all_errors = OrderedDict((p, self.params[p][error_name]) for p in params)

        return all_errors, new_result

    def _error(self, params, method, sigma):
        if not callable(method):
            try:
                method = self._error_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))
        return method(result=self, params=params, sigma=sigma)

    def covariance(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, as_dict: bool = False):
        """Calculate the covariance matrix for `params`.

            Args:
                params (list(:py:class:`~zfit.Parameter`)): The parameters to calculate
                    the covariance matrix. If `params` is `None`, use all *floating* parameters.
                method (str or Callbel): The method to use to calculate the covariance matrix. Valid choices are
                    {'minuit_hesse', 'hesse_np'} or a Callable.
                as_dict (bool): Default `False`. If `True` then returns a dictionnary.

            Returns:
                2D `numpy.array` of shape (N, N);
                `dict`(param1, param2) -> covariance if `as_dict == True`.
        """
        if method is None:
            # LEGACY START
            method = self._default_hesse
            from zfit.minimizers.minimizer_minuit import Minuit

            if isinstance(self.minimizer, Minuit):
                method = "minuit_hesse"
            # LEGACY END

        params = self._input_check_params(params)
        covariance = self._covariance(method=method)
        covariance = {k: covariance[k] for k in itertools.product(params, params)}

        if as_dict:
            return covariance
        else:
            return dict_to_matrix(params, covariance)

    def _covariance(self, method):
        if not callable(method):
            try:
                method = self._hesse_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))

        params = list(self.params.keys())
        return method(result=self, params=params)

    def __str__(self):
        string = Style.BRIGHT + f'FitResult' + Style.NORMAL + f' of\n{self.loss} \nwith\n{self.minimizer}\n'
        string += tafo.generate_table(
            [[color_on_bool(self.converged), format_value(self.edm, highprec=False),
              format_value(self.fmin)]],
            ['converged', 'edm', 'min value'],
            # grid_style=tafo.SparseGrid()
        )
        string += Style.BRIGHT + "Parameters\n"
        string += str(self.params)
        return string

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(self.__repr__())
            return
        p.text(self.__str__())


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


def format_value(value, highprec=True):
    try:
        import iminuit
        m_error_class = iminuit.util.MError
    except ImportError:
        m_error_class = dict
    if isinstance(value, (dict, m_error_class)):
        if 'error' in value:
            value = value['error']
            value = f"+/- {value:> 6.2g}"
        if 'lower' in value and 'upper' in value:
            lower = value['lower']
            upper = value['upper']
            lower, upper = f"{lower: >+6.2g}", f"{upper: >+6.2g}"
            lower += " " * (9 - len(lower))
            value = lower + upper

    if isinstance(value, float):
        if highprec:
            value = f"{value:> 6.4g}"
        else:
            value = f"{value:> 6.2g}"
    return value


def color_on_bool(value, on_true=colored.bg(10), on_false=colored.bg(9)):
    if not value and on_false:
        value_add = on_false
    elif value and on_true:
        value_add = on_true
    else:
        value_add = ''
    value = value_add + str(value)
    return value


class ParamHolder(dict):  # no UserDict, we only want to change the __str__

    def __str__(self) -> str:
        order_keys = ['value', 'hesse']
        keys = OrderedSet()
        for pdict in self.values():
            keys.update(OrderedSet(pdict))
        order_keys = OrderedSet([key for key in order_keys if key in keys])
        order_keys.update(keys)

        rows = []
        for param, pdict in self.items():
            row = [param.name]
            row.extend(format_value(pdict.get(key, ' ')) for key in order_keys)
            row.append(color_on_bool(run(param.at_limit), on_true=colored.bg('light_red'), on_false=False))
            rows.append(row)

        order_keys = ['name'] + list(order_keys) + ['at limit']

        return tafo.generate_table(rows, order_keys,
                                   grid_style=tafo.AlternatingRowGrid(colored.bg(15), colored.bg(254)))

#  Copyright (c) 2021 zfit
import contextlib
import itertools
import warnings
from collections import OrderedDict
from typing import Dict, Union, Callable, Optional, Tuple, Iterable

import colored
import iminuit
import numpy as np
import scipy.stats
from colorama import Style, init
from ordered_set import OrderedSet
from scipy.optimize import LbfgsInvHessProduct
from tabulate import tabulate
import scipy.optimize

from .errors import compute_errors, covariance_with_weights, dict_to_matrix, matrix_to_dict
from .interface import ZfitMinimizer, ZfitResult
from .termination import ConvergenceCriterion
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..core.parameter import set_values
from ..settings import run
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_args
from ..util.warnings import ExperimentalFeatureWarning
from ..util.ztyping import ParamsTypeOpt

init(autoreset=True)


def _minos_minuit(result, params, cl=None, *, minuit_minimizer=None):
    from zfit.minimize import Minuit

    fitresult = result
    minimizer = fitresult.minimizer
    if not isinstance(minimizer, Minuit):
        if minuit_minimizer is None:
            raise TypeError("To use `minos_error` method with a minimizer other than `Minuit`, a Minuit minimizer"
                            " needs to be passed to the method (it doesn't has to be minimized though).")
        minimizer = minuit_minimizer

    merror_result = minimizer._minuit_minimizer.minos(*(p.name for p in params), cl=cl).merrors  # returns every var
    attrs = ['lower', 'upper', 'is_valid', 'upper_valid', 'lower_valid', 'at_lower_limit', 'at_upper_limit', 'nfcn']
    result = {}
    for p, merror in zip(params, merror_result.values()):
        result[p] = {attr: getattr(merror, attr) for attr in attrs}
        result[p]['original'] = result
    # result = OrderedDict((p, result[p.name]) for p in params)
    new_result = None
    return result, new_result


def _covariance_minuit(result, params):
    if any(data.weights is not None for data in result.loss.data):
        warnings.warn("The computation of the covariance matrix with weights is still experimental.",
                      ExperimentalFeatureWarning)

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
    if any(data.weights is not None for data in result.loss.data):
        warnings.warn("The computation of the covariance matrix with weights is still experimental.",
                      ExperimentalFeatureWarning)

    # TODO: maybe activate again? currently fails due to numerical problems
    # numgrad_was_none = settings.options.numerical_grad is None
    # if numgrad_was_none:
    #     settings.options.numerical_grad = True

    _, gradient, hessian = result.loss.value_gradients_hessian(params)
    covariance = np.linalg.inv(hessian)

    # if numgrad_was_none:
    #     settings.options.numerical_grad = None

    return matrix_to_dict(params, covariance)


class FitResult(ZfitResult):
    _default_hesse = "hesse_np"
    _hesse_methods = {"minuit_hesse": _covariance_minuit, "hesse_np": _covariance_np}
    _default_error = "zfit_error"
    _error_methods = {"minuit_minos": _minos_minuit, "zfit_error": compute_errors}

    def __init__(self, params: Dict[ZfitParameter, float], edm: float, fmin: float, status: int, converged: bool,
                 info: dict, loss: ZfitLoss, minimizer: "ZfitMinimizer",
                 criterion: Optional[ConvergenceCriterion] = None):
        """Create a `FitResult` from a minimization. Store parameter values, minimization infos and calculate errors.

        Any errors calculated are saved under `self.params` dictionary with::

            {parameter: {error_name1: {'low': value, 'high': value or similar}}

        Args:
            params: Result of the fit where each
               :py:class:`~zfit.Parameter` key has the value from the minimum found by the minimizer.
            edm: The estimated distance to minimum, estimated by the minimizer (if available)
            fmin: The minimum of the function found by the minimizer
            status: A status code (if available)
            converged: Whether the fit has successfully converged or not.
            info: Additional information (if available) like *number of function calls* and the
                original minimizer return message.
            loss: The loss function that was minimized. Contains also the pdf, data etc.
            minimizer: Minimizer that was used to obtain this `FitResult` and will be used to
                calculate certain errors. If the minimizer is state-based (like "iminuit"), then this is a copy
                and the state of other `FitResults` or of the *actual* minimizer that performed the minimization
                won't be altered.
        """
        super().__init__()

        self._status = status
        self._converged = converged
        self._params = self._input_convert_params(params)
        self._params_at_limit = any(param.at_limit for param in self.params)
        self._edm = edm
        self._fmin = fmin
        self._info = info
        self._loss = loss
        self._minimizer = minimizer
        self._valid = True
        self._covariance_dict = {}

    def _input_convert_params(self, params):
        return ParamHolder((p, {"value": v}) for p, v in params.items())

    def _get_uncached_params(self, params, method_name):
        return [p for p in params if self.params[p].get(method_name) is None]

    @classmethod
    def from_minuit(cls, loss: ZfitLoss, params: Iterable[ZfitParameter], result: iminuit.util.FMin,
                    minimizer: Union[ZfitMinimizer, iminuit.Minuit]) -> 'FitResult':
        """Create a `FitResult` from a :py:class:~`iminuit.util.MigradResult` returned by
        :py:meth:`iminuit.Minuit.migrad` and a iminuit :py:class:~`iminuit.Minuit` instance with the corresponding
        zfit objects.

        Args:
            loss: zfit Loss that was minimized.
            params: Iterable of the zfit parameters that were floating during the minimization.
            result: Return value of the iminuit migrad command.
            minimizer: Instance of the iminuit Minuit that was used to minimize the loss.

        Returns:
            A `FitResult` as if zfit Minuit was used.
        """

        from .minimizer_minuit import Minuit
        if not isinstance(minimizer, Minuit):
            if isinstance(minimizer, iminuit.Minuit):
                minimizer_new = Minuit()
                minimizer_new._minuit_minimizer = minimizer
                minimizer = minimizer_new
            else:
                raise ValueError(f"Minimizer {minimizer} not supported. Use `Minuit` from zfit or from iminuit.")
        params_result = [p_dict for p_dict in result.params]
        result_vals = [res.value for res in params_result]
        set_values(params, values=result_vals)
        fmin_object = result.fmin
        info = {'n_eval': fmin_object.nfcn,
                'n_iter': "REMOVED FROM MINUIT",  # TODO 6.x: remove completely
                # 'grad': result['jac'],
                # 'message': result['message'],
                'original': fmin_object}
        edm = fmin_object.edm
        fmin = fmin_object.fval
        status = -999
        converged = fmin_object.is_valid
        params = OrderedDict((p, res.value) for p, res in zip(params, params_result))
        return cls(params=params, edm=edm, fmin=fmin, info=info, loss=loss,
                   status=status, converged=converged,
                   minimizer=minimizer)

    @classmethod
    def from_scipy(cls, loss: ZfitLoss, params: Iterable[ZfitParameter], result: scipy.optimize.OptimizeResult,
                   minimizer: ZfitMinimizer, edm=False, valid=None, criterion=None):
        result_values = result['x']
        converged = result['success']
        status = result['status']
        inv_hesse = result.get('hess_inv')
        if isinstance(inv_hesse, LbfgsInvHessProduct):
            inv_hesse = inv_hesse.todense()
        info = {'n_eval': result['nfev'],
                'n_iter': result['nit'],
                'grad': result.get('jac'),
                'inv_hesse': inv_hesse,
                'message': result['message'],
                'original': result}
        fmin = result['fun']
        params = OrderedDict((p, v) for p, v in zip(params, result_values))

        fitresult = cls(params=params, edm=edm, fmin=fmin, info=info,
                              converged=converged, status=status,
                              loss=loss, minimizer=minimizer, criterion=criterion)
        if isinstance(valid, str):
            fitresult._valid = False
            fitresult.info['invalid_message'] = valid
        return fitresult

    @classmethod
    def from_nlopt(cls, loss, minimizer, opt, edm, params, xvalues, n_eval=None,
                   inv_hess=None, valid=None, criterion=None):
        param_dict = {p: v for p, v in zip(params, xvalues)}
        fmin = opt.last_optimum_value()
        status = opt.last_optimize_result()
        n_eval = opt.get_numevals() if n_eval is None else n_eval
        converged = 1 <= status <= 4
        messages = {
            1: "NLOPT_SUCCESS",
            2: "NLOPT_STOPVAL_REACHED",
            3: "NLOPT_FTOL_REACHED",
            4: "NLOPT_XTOL_REACHED",
            5: "NLOPT_MAXEVAL_REACHED",
            6: "NLOPT_MAXTIME_REACHED",
            -1: "NLOPT_FAILURE",
            -2: "NLOPT_INVALID_ARGS",
            -3: "NLOPT_OUT_OF_MEMORY",
            -4: "NLOPT_ROUNDOFF_LIMITED",
            -5: "NLOPT_FORCED_STOP",
        }
        message = messages[status]
        info = {'n_eval': n_eval,
                'message': message,
                'original': status,
                'status': status}
        if inv_hess is not None:
            info['inv_hess'] = inv_hess

        result = cls(params=param_dict, edm=edm, fmin=fmin, status=status, converged=converged, info=info,
                           loss=loss, minimizer=minimizer)
        if valid:
            result._valid = False
            result.info['invalid_message'] = valid
        return result

    @property
    def params(self):
        return self._params

    @property
    def edm(self):
        """The estimated distance to the minimum.

        Returns:
            Numeric
        """
        return self._edm

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
            Numeric
        """
        return self._fmin

    @property
    def status(self):
        return self._status

    @property
    def info(self):
        return self._info

    @property
    def converged(self):
        return self._converged

    @property
    def valid(self):
        return self._valid and not self.params_at_limit and self.converged

    @property
    def params_at_limit(self):
        return self._params_at_limit

    @contextlib.contextmanager
    def _input_check_reset_params(self, params):
        params = self._input_check_params(params=params)
        old_values = run(params)
        try:
            yield params
        except Exception:
            warnings.warn("Exception occurred, parameter values are not reset and in an arbitrary, last"
                          " used state. If this happens during normal operation, make sure you reset the values.",
                          RuntimeWarning)
            raise
        set_values(params=params, values=old_values)

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
            params: The parameters to calculate the
                Hessian symmetric error. If None, use all parameters.
            method: the method to calculate the covariance matrix. Can be {'minuit_hesse', 'hesse_np'} or a callable.
            error_name: The name for the error in the dictionary.

        Returns:
            Result of the hessian (symmetric) error as dict with each parameter holding
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

        with self._input_check_reset_params(params) as params:
            uncached_params = self._get_uncached_params(params=params, method_name=error_name)
            if uncached_params:
                error_dict = self._hesse(params=uncached_params, method=method)
                self._cache_errors(error_name=error_name, errors=error_dict)

        errors = OrderedDict((p, self.params[p][error_name]) for p in params)
        return errors

    def _cache_errors(self, error_name, errors):
        for param, error in errors.items():
            self.params[param][error_name] = error

    def _hesse(self, params, method):
        covariance_dict = self.covariance(params, method, as_dict=True)
        return OrderedDict((p, {"error": covariance_dict[(p, p)] ** 0.5}) for p in params)

    def error(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, error_name: str = None,
              sigma: float = 1.0) -> OrderedDict:
        r"""

        .. deprecated:: unknown
            Use :func:`errors` instead.

        Args:
            params: The parameters or their names to calculate the
                 errors. If `params` is `None`, use all *floating* parameters.
            method: The method to use to calculate the errors. Valid choices are
                {'minuit_minos'} or a Callable.
            sigma: Errors are calculated with respect to `sigma` std deviations. The definition
                of 1 sigma depends on the loss function and is defined there.

                For example, the negative log-likelihood (without the factor of 2) has a correspondents
                of :math:`\Delta` NLL of 1 corresponds to 1 std deviation.
            error_name: The name for the error in the dictionary.


        Returns:
            A `OrderedDict` containing as keys the parameter names and as value a `dict` which
                contains (next to probably more things) two keys 'lower' and 'upper',
                holding the calculated errors.
                Example: result['par1']['upper'] -> the asymmetric upper error of 'par1'
        """
        warnings.warn("`error` is deprecated, use `errors` instead. This will return not only the errors but also "
                      "(a possible) new FitResult if a minimum was found. So change"
                      "errors = result.error()"
                      "to"
                      "errors, new_res = result.errors()", DeprecationWarning)
        return self.errors(params=params, method=method, error_name=error_name, sigma=sigma)[0]

    @deprecated_args(None, "Use cl for confidence level instead.", 'sigma')
    def errors(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, error_name: str = None,
               cl: Optional[float] = None, sigma=None) -> Tuple[OrderedDict, Union[None, 'FitResult']]:
        r"""Calculate and set for `params` the asymmetric error using the set error method.

            Args:
                params: The parameters or their names to calculate the
                     errors. If `params` is `None`, use all *floating* parameters.
                method: The method to use to calculate the errors. Valid choices are
                    {'minuit_minos'} or a Callable.
                cl: Uncertainties are calculated with respect to the confidence level cl. The default is 68.3%.
                    For example, the negative log-likelihood (without the factor of 2) has a correspondents
                    of :math:`\Delta` NLL of 1 corresponds to 1 std deviation.
                sigma: Errors are calculated with respect to `sigma` std deviations. The definition
                    of 1 sigma depends on the loss function and is defined there.
                error_name: The name for the error in the dictionary.


            Returns:
                A `OrderedDict` containing as keys the parameter and as value a `dict` which
                    contains (next to probably more things) two keys 'lower' and 'upper',
                    holding the calculated errors.
                    Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        """
        if sigma is not None:
            if cl is not None:
                raise ValueError("Cannot define sigma and cl, use cl only.")
            else:
                cl = scipy.stats.chi2(1).ppf(cl)

        if cl is None:
            cl = 0.68268949  # scipy.stats.chi2(1).cdf(1)

        if method is None:
            # TODO: legacy, remove 0.6
            from zfit.minimize import Minuit
            if isinstance(self.minimizer, Minuit):
                method = 'minuit_minos'
                warnings.warn("'minuit_minos' will be changed as the default errors method to a custom implementation"
                              "with the same functionality. If you want to make sure that 'minuit_minos' will be used "
                              "in the future, add it explicitly as in `errors(method='minuit_minos')`", FutureWarning)
            else:
                method = self._default_error
        if error_name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            error_name = method

        if method == 'zfit_error':
            warnings.warn("'zfit_error' is still experimental and may fails.", ExperimentalFeatureWarning)

        params = self._input_check_params(params)

        with self._input_check_reset_params(self.params.keys()):
            # TODO: cache with cl!
            uncached_params = self._get_uncached_params(params=params, method_name=error_name)

            new_result = None

            if uncached_params:
                error_dict, new_result = self._error(params=uncached_params, method=method, cl=cl)
                if new_result is None:
                    self._cache_errors(error_name=error_name, errors=error_dict)
                else:
                    msg = "Invalid, a new minimum was found."
                    self._cache_errors(error_name=error_name, errors={p: msg for p in params})
                    self._valid = False
                    new_result._cache_errors(error_name=error_name, errors=error_dict)
        all_errors = OrderedDict((p, self.params[p][error_name]) for p in params)

        return all_errors, new_result

    def _error(self, params, method, cl):
        if not callable(method):
            try:
                method = self._error_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))
        return method(result=self, params=params, cl=cl)

    def covariance(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, as_dict: bool = False):
        """Calculate the covariance matrix for `params`.

            Args:
                params: The parameters to calculate
                    the covariance matrix. If `params` is `None`, use all *floating* parameters.
                method: The method to use to calculate the covariance matrix. Valid choices are
                    {'minuit_hesse', 'hesse_np'} or a Callable.
                as_dict: Default `False`. If `True` then returns a dictionnary.

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

        if method not in self._covariance_dict:
            with self._input_check_reset_params(params) as params:
                self._covariance_dict[method] = self._covariance(method=method)

        params = self._input_check_params(params)
        covariance = {k: self._covariance_dict[method][k] for k in itertools.product(params, params)}

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

        if any(data.weights is not None for data in self.loss.data):
            return covariance_with_weights(method=method, result=self, params=params)
        else:
            return method(result=self, params=params)

    def correlation(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, as_dict: bool = False):
        """Calculate the correlation matrix for `params`.

            Args:
                params: The parameters to calculate
                    the correlation matrix. If `params` is `None`, use all *floating* parameters.
                method: The method to use to calculate the correlation matrix. Valid choices are
                    {'minuit_hesse', 'hesse_np'} or a Callable.
                as_dict: Default `False`. If `True` then returns a dictionnary.

            Returns:
                2D `numpy.array` of shape (N, N);
                `dict`(param1, param2) -> correlation if `as_dict == True`.
        """

        covariance = self.covariance(params=params, method=method, as_dict=False)
        correlation = covariance_to_correlation(covariance)

        if as_dict:
            params = self._input_check_params(params)
            return matrix_to_dict(params, correlation)
        else:
            return correlation

    def __str__(self):
        string = Style.BRIGHT + f'FitResult' + Style.NORMAL + f' of\n{self.loss} \nwith\n{self.minimizer}\n\n'
        string += tabulate(
            [[color_on_bool(self.valid), color_on_bool(self.converged, on_true=False),
              color_on_bool(self.params_at_limit, on_true=colored.bg(9), on_false=False),
              format_value(self.edm, highprec=False),
              format_value(self.fmin)]],
            ['valid', 'converged', 'param at limit', 'edm', 'min value'],
            tablefmt='fancy_grid',
            disable_numparse=True)
        string += '\n\n' + Style.BRIGHT + "Parameters\n" + Style.NORMAL
        string += str(self.params)
        return string

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(self.__repr__())
            return
        p.text(self.__str__())


def covariance_to_correlation(covariance):
    diag = np.diag(1 / np.diag(covariance) ** 0.5)
    return np.matmul(diag, np.matmul(covariance, diag))


def format_value(value, highprec=True):
    try:
        import iminuit
        m_error_class = iminuit.util.MError
    except ImportError:
        m_error_class = dict

    if isinstance(value, dict) and 'error' in value:
        value = value['error']
        value = f"{value:> 6.2g}"
        value = f'+/-{" " * (8 - len(value))}' + value
    if isinstance(value, m_error_class) or (isinstance(value, dict) and 'lower' in value and 'upper' in value):
        if isinstance(value, m_error_class):
            lower = value.lower
            upper = value.upper
        else:
            lower = value['lower']
            upper = value['upper']
        lower_sign = f"{np.sign(lower): >+}"[0]
        upper_sign = f"{np.sign(upper): >+}"[0]
        lower, upper = f"{np.abs(lower): >6.2g}", f"{upper: >6.2g}"
        lower = lower_sign + " " * (7 - len(lower)) + lower
        upper = upper_sign + " " * (7 - len(upper)) + upper
        # lower += " t" * (11 - len(lower))
        value = lower + " " * 3 + upper

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
    value = value_add + str(value) + Style.RESET_ALL
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
        table = tabulate(rows, order_keys, numalign="right", stralign='right', colalign=('left',))
        return table



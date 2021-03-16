#  Copyright (c) 2021 zfit
import collections
import contextlib
import itertools
import warnings
from collections import OrderedDict
from typing import Dict, Union, Callable, Optional, Tuple, Iterable, Mapping, List

import colored
import iminuit
import numpy as np
import scipy.optimize
import scipy.stats
from colorama import Style, init
from ordered_set import OrderedSet
from scipy.optimize import LbfgsInvHessProduct
from tabulate import tabulate

from .errors import compute_errors, covariance_with_weights, dict_to_matrix, matrix_to_dict
from .interface import ZfitMinimizer, ZfitResult
from .termination import ConvergenceCriterion
from ..core.interfaces import ZfitLoss, ZfitParameter, ZfitIndependentParameter
from ..core.parameter import set_values
from ..settings import run
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_args
from ..util.warnings import ExperimentalFeatureWarning
from ..util.ztyping import ParamsTypeOpt

init(autoreset=True)


class Approximations:

    def __init__(self, params: List[ZfitParameter], gradient: Optional[np.ndarray] = None,
                 hessian: Optional[np.ndarray] = None,
                 inv_hessian: Optional[np.ndarray] = None) -> None:
        """Holds different approximations after the minimisation and/or calculates them.

        Args:
            params: List of parameters the approximations (gradient, hessian, ...) were calculated with.
            gradient: Gradient
            hessian: Hessian Matrix
            inv_hessian: Inverse of the Hessian Matrix
        """
        self._params = params
        self._gradient = gradient
        self._hessian = hessian
        self._inv_hessian = inv_hessian
        super().__init__()

    @property
    def params(self):
        return self._params

    def gradient(self,
                 params: Optional[Union[ZfitParameter, Iterable[ZfitParameter]]] = None
                 ) -> Union[np.ndarray, None]:
        """Return an approximation of the gradient _if available_.

        Args:
            params: Parameters to which the gradients should be returned

        Returns:
            Array with gradients or `None`
        """
        grad = self._gradient
        if grad is None:
            return None

        if params is not None:
            params = convert_to_container(params, container=tuple)
            params_mapped = {i: params.index(param) for i, param in enumerate(self.params) if param in params}
            indices = sorted(params_mapped, key=lambda x: params_mapped[x])
            grad = grad[indices]
        return grad

    def hessian(self, invert: bool = True) -> Union[np.ndarray, None]:
        """Return an approximation of the hessian _if available_.

        Args:
            invert: If a _hessian approximation_ is not available but an inverse hessian is, invert the latter to
                obtain the hessian approximation.

        Returns:
            Array with hessian matrix or `None`
        """
        hess = self._hessian
        if hess is None and invert:
            inv_hess = self._inv_hessian
            if inv_hess is not None:
                hess = np.linalg.inv(inv_hess)
                self._hessian = hess
        return hess

    def inv_hessian(self, invert: bool = True) -> Union[None, np.ndarray]:
        """Return an approximation of the inverse hessian _if available_.

        Args:
            invert: If an _inverse hessian approximation_ is not available but a hessian is, invert the latter to
                obtain the inverse hessian approximation.

        Returns:
            Array with the inverse of the hessian matrix or `None`
        """
        inv_hess = self._inv_hessian
        if inv_hess is None and invert:
            hess = self._hessian
            if hess is not None:
                inv_hess = np.linalg.inv(hess)
                self._inv_hessian = inv_hess
        return inv_hess


def _minos_minuit(result, params, cl=None):
    minuit_minimizer = result._create_minuit_instance()

    merror_result = minuit_minimizer.minos(*(p.name for p in params), cl=cl).merrors  # returns every var
    attrs = ['lower', 'upper', 'is_valid', 'upper_valid', 'lower_valid', 'at_lower_limit', 'at_upper_limit', 'nfcn']
    errors = {}
    for p in params:
        error_res = merror_result[p.name]
        errors[p] = {attr: getattr(error_res, attr) for attr in attrs}
        errors[p]['original'] = error_res
    new_result = None
    return errors, new_result


def _covariance_minuit(result, params):
    if any(data.weights is not None for data in result.loss.data):
        warnings.warn("The computation of the covariance matrix with weights is still experimental.",
                      ExperimentalFeatureWarning)

    minuit_minimizer = result._create_minuit_instance()

    _ = minuit_minimizer.hesse()  # make sure to have an accurate covariance

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

    _, gradient, hessian = result.loss.value_gradient_hessian(params)
    covariance = np.linalg.inv(hessian)

    # if numgrad_was_none:
    #     settings.options.numerical_grad = None

    return matrix_to_dict(params, covariance)


def _covariance_approx(result, params):
    if any(data.weights is not None for data in result.loss.data):
        warnings.warn("Approximate covariance/hesse estimation with weights is not supported, returning None",
                      RuntimeWarning)
    covariance_dict = {}
    inv_hessian = result.approx.inv_hessian(invert=True)
    if inv_hessian is None:
        return covariance_dict

    params_approx = list(result.params)
    for p1 in params:
        p1_index = params_approx.index(p1)
        for p2 in params:
            p2_index = params_approx.index(p2)
            index = (p1_index, p2_index)
            key = (p1, p2)
            covariance_dict[key] = inv_hessian[index]
    return covariance_dict


class ParamToNameGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            item = item.name
        return super().__getitem__(item)


class NameToParamGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, str):
            for param in self.keys():
                if param.name == item:
                    item = param
                    break
        return super().__getitem__(item)


class FitResult(ZfitResult):
    _default_hesse = "hesse_np"
    _hesse_methods = {"minuit_hesse": _covariance_minuit, "hesse_np": _covariance_np, "approx": _covariance_approx}
    _default_error = "zfit_error"
    _error_methods = {"minuit_minos": _minos_minuit, "zfit_error": compute_errors}

    def __init__(self,
                 params: Dict[ZfitParameter, float],
                 edm: float,
                 fmin: float,
                 loss: ZfitLoss,
                 minimizer: "ZfitMinimizer",
                 valid: bool,
                 criterion: Optional[ConvergenceCriterion],
                 status: Optional[int] = None,
                 converged: Optional[bool] = None,
                 message: Optional[str] = None,
                 info: Optional[Mapping] = None,
                 approx: Optional[Union[Mapping, Approximations]] = None,
                 niter: Optional[int] = None,
                 evaluator: "zfit.minimizer.evaluation.LossEval" = None,
                 ) -> None:
        """Create a `FitResult` from a minimization. Store parameter values, minimization infos and calculate errors.

        Any errors calculated are saved under `self.params` dictionary with::

            {parameter: {error_name1: {'low': value, 'high': value or similar}}

        Args:
            valid (Union[bool, None, numpy.bool_, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]):
            criterion (Union[zfit.minimizers.termination.EDM, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]):
            message (str):
            approx (Union[None, None, Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], None, Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]], None, None, None, Dict[str, List[zfit.core.parameter.Parameter]], Dict[str, List[zfit.core.parameter.Parameter]]]):
            niter (Union[int, None, None]):
            evaluator (Union[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, zfit.minimizers.evaluation.LossEval, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]):
            params: Result of the fit where each
               :py:class:`~zfit.Parameter` key has the value from the minimum found by the minimizer.
            edm: The estimated distance to minimum, estimated by the minimizer (if available)
            fmin: Value at the minimum of the function found by the minimizer
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

        if status is None:
            status = 0 if valid else -999
        if converged is None and valid:
            converged = True
        if message is None:
            if valid:
                message = ''
            else:
                message = "Invalid, unknown reason (not specified in init)"

        info = {} if info is None else info
        approx = self._input_convert_approx(approx, evaluator, info, params)

        if evaluator is not None:
            niter = evaluator.niter if niter is None else niter

        self._cache_minuit = None  # in case used in errors

        self._evaluator = evaluator  # keep private for now
        self._niter = niter  # keep private for now
        self._approx = approx
        self._status = status
        self._message = "" if message is None else message
        self._converged = converged
        self._params = self._input_convert_params(params)
        self._values = ValuesHolder(params)
        self._params_at_limit = any(param.at_limit for param in self.params)
        self._edm = edm
        self._criterion = criterion
        self._fmin = fmin
        self._info = info
        self._loss = loss
        self._minimizer = minimizer
        self._valid = valid
        self._covariance_dict = {}

    def _input_convert_approx(self, approx, evaluator, info, params):
        """Convert approx (if a Mapping) to an `Approximation` using the information provided.

        Args:
            approx:
            evaluator:
            info:
            params:

        Returns:
            The created approximation.
        """
        approx = {} if approx is None else approx
        if isinstance(approx, collections.Mapping):
            if 'params' not in approx:
                approx['params'] = params

            if info:
                if 'gradient' not in approx:
                    approx['gradient'] = info.get('grad', info.get('gradient'))
                if 'hessian' not in approx:
                    approx['hessian'] = info.get('hess', info.get('hesse', info.get('hessian')))
                if 'inv_hessian' not in approx:
                    approx['inv_hessian'] = info.get('inv_hess', info.get('inv_hesse', info.get('inv_hessian')))
            if evaluator is not None:
                if 'gradient' not in approx:
                    approx['gradient'] = evaluator.last_gradient
                if 'hessian' not in approx:
                    approx['hessian'] = evaluator.last_hessian

            approx = Approximations(**approx)
        return approx

    def _input_convert_params(self, params):
        return ParamHolder((p, {"value": v}) for p, v in params.items())

    def _get_uncached_params(self, params, method_name):
        # TODO: Also cache sigma!
        return [p for p in params if self.params[p].get(method_name) is None]

    def _create_minuit_instance(self):
        minuit = self._cache_minuit
        from zfit.minimizers.minimizer_minuit import Minuit
        if minuit is None:
            if isinstance(self.minimizer, Minuit):
                minuit = self.minimizer._minuit_minimizer
            else:
                minimizer = Minuit(tol=self.minimizer.tol, verbosity=0, name="ZFIT_TMP_UNCERTAINTY")
                minuit, _, _ = minimizer._make_minuit(loss=self.loss, params=self.params, init=self)
            self._cache_minuit = minuit
        return minuit

    @classmethod
    def from_ipopt(cls, loss: ZfitLoss,
                   params: Iterable[ZfitParameter],
                   opt_instance: 'ipyopt.Problem',
                   minimizer: 'zfit.minimize.IpyoptV1',
                   converged: Optional[bool],
                   xvalues: np.ndarray,
                   message: Optional[str],
                   edm: Union['zfit.minimizers.termination.CriterionNotAvailable', float],
                   niter: Optional[int],
                   valid: bool,
                   criterion: 'zfit.minimizers.termination.ConvergenceCriterion',
                   evaluator: Optional['zfit.minimizers.evaluation.LossEval'],
                   fmin: Optional[float],
                   status: Optional[int]
                   ) -> 'FitResult':

        info = {'original_optimizer': opt_instance}
        params = dict((p, val) for p, val in zip(params, xvalues))
        return cls(params=params, loss=loss, fmin=fmin, edm=edm, message=message,
                   criterion=criterion, info=info, valid=valid, converged=converged,
                   niter=niter, status=status, minimizer=minimizer, evaluator=evaluator)

    @classmethod
    def from_minuit(cls, loss: ZfitLoss, params: Iterable[ZfitParameter], minuit_opt: iminuit.util.FMin,
                    minimizer: Union[ZfitMinimizer, iminuit.Minuit], valid, message, criterion=None) -> 'FitResult':
        """Create a `FitResult` from a :py:class:~`iminuit.util.MigradResult` returned by
        :py:meth:`iminuit.Minuit.migrad` and a iminuit :py:class:~`iminuit.Minuit` instance with the corresponding
        zfit objects.

        Args:
            loss: zfit Loss that was minimized.
            params: Iterable of the zfit parameters that were floating during the minimization.
            minuit_opt: Return value of the iminuit migrad command.
            minimizer: Instance of the iminuit Minuit that was used to minimize the loss.

        Returns:
            A `FitResult` as if zfit Minuit was used.
        """
        from .termination import EDM
        from .minimizer_minuit import Minuit

        if not isinstance(minimizer, Minuit):
            if isinstance(minimizer, iminuit.Minuit):
                minimizer_new = Minuit()
                minimizer_new._minuit_minimizer = minimizer
                minimizer = minimizer_new
            else:
                raise ValueError(f"Minimizer {minimizer} not supported. Use `Minuit` from zfit or from iminuit.")

        params_result = [p_dict for p_dict in minuit_opt.params]

        fmin_object = minuit_opt.fmin

        info = {'n_eval': fmin_object.nfcn,
                # 'grad': result['jac'],
                # 'message': result['message'],
                'original_minimizer': minuit_opt,
                'original': fmin_object}
        if fmin_object.has_covariance:
            info['inv_hessian'] = np.array(minuit_opt.covariance)
        edm = fmin_object.edm
        if criterion is None:
            criterion = EDM(tol=minimizer.tol, loss=loss, params=params)
            criterion.last_value = edm
        fmin = fmin_object.fval
        status = -999
        valid = valid and fmin_object.is_valid
        converged = fmin_object.is_valid
        params = dict((p, res.value) for p, res in zip(params, params_result))
        return cls(params=params, edm=edm, fmin=fmin, info=info, loss=loss, niter=fmin_object.nfcn,
                   status=status, converged=converged, message=message, valid=valid, criterion=criterion,
                   minimizer=minimizer)

    @classmethod
    def from_scipy(cls, loss: ZfitLoss, params: Iterable[ZfitParameter], result: scipy.optimize.OptimizeResult,
                   minimizer: ZfitMinimizer, message=None, edm=False, niter=None, valid=None, criterion=None,
                   evaluator=None) -> 'FitResult':

        result_values = result['x']
        if niter is None:
            niter = result.get('nit')

        converged = result.get('success', valid)
        status = result['status']
        inv_hesse = result.get('hess_inv')
        if isinstance(inv_hesse, LbfgsInvHessProduct):
            inv_hesse = inv_hesse.todense()
        grad = result.get('grad')
        if message is None:
            message = result.get('message')
        info = {'n_eval': result['nfev'],
                'n_iter': niter,
                'niter': niter,
                'grad': result.get('jac') if grad is None else grad,
                'inv_hesse': inv_hesse,
                'hesse': result.get('hesse'),
                'message': message,
                'evaluator': evaluator,
                'original': result}
        approx = dict(params=params,
                      gradient=info.get('grad'),
                      hessian=info.get('hesse'),
                      inv_hessian=info.get('inv_hesse'))

        fmin = result['fun']
        params = dict((p, v) for p, v in zip(params, result_values))

        fitresult = cls(params=params, edm=edm, fmin=fmin, info=info, approx=approx,
                        converged=converged, status=status, message=message, valid=valid, niter=niter,
                        loss=loss, minimizer=minimizer, criterion=criterion, evaluator=evaluator)
        if isinstance(valid, str):
            fitresult._valid = False
            fitresult.info['invalid_message'] = valid
        return fitresult

    @classmethod
    def from_nlopt(cls, loss, minimizer, opt, edm, params, xvalues, message,
                   valid, criterion, evaluator,
                   niter=None,
                   inv_hess=None, hess=None,
                   ):
        param_dict = {p: v for p, v in zip(params, xvalues)}
        fmin = opt.last_optimum_value()
        status = opt.last_optimize_result()
        niter = opt.get_numevals() if niter is None else niter
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
        message_nlopt = messages[status]
        info = {'n_eval': niter,
                'niter': niter,
                'message': message_nlopt,
                'original': status,
                'evaluator': evaluator,
                'status': status}
        if message is None:
            message = message_nlopt

        approx = {}
        if inv_hess is None:
            if hess is None and evaluator is not None:
                hess = evaluator.last_hessian
            if hess is not None:
                inv_hess = np.linalg.inv(hess)

        if inv_hess is not None:
            info['inv_hesse'] = inv_hess
            approx['inv_hessian'] = inv_hess

        result = cls(params=param_dict,
                     edm=edm,
                     fmin=fmin,
                     status=status,
                     converged=converged,
                     info=info,
                     niter=niter,
                     valid=valid,
                     loss=loss,
                     minimizer=minimizer,
                     criterion=criterion,
                     message=message,
                     evaluator=evaluator)
        return result

    @property
    def approx(self) -> Approximations:
        return self._approx

    @property
    def params(self) -> Mapping[ZfitIndependentParameter, Mapping[str, Mapping[str, object]]]:
        return self._params

    @property
    def values(self) -> Mapping[Union[str, ZfitParameter], float]:
        return self._values

    @property
    def criterion(self) -> ConvergenceCriterion:
        return self._criterion

    @property
    def message(self) -> str:
        return self._message

    @property
    def edm(self) -> float:
        """The estimated distance to the minimum.

        Returns:
            Numeric
        """
        return self._edm

    @property
    def minimizer(self) -> ZfitMinimizer:
        return self._minimizer

    @property
    def loss(self) -> ZfitLoss:
        # TODO(Mayou36): this is currently a reference, should be a copy of the loss?
        return self._loss

    @property
    def fmin(self) -> float:
        """Function value at the minimum.

        Returns:
            Numeric
        """
        return self._fmin

    @property
    def status(self):
        return self._status

    @property
    def info(self) -> Mapping[str, object]:
        return self._info

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def valid(self) -> bool:
        return self._valid and not self.params_at_limit and self.converged

    @property
    def params_at_limit(self) -> bool:
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

    @deprecated_args(None, "Use `name` instead", "error_name")
    def hesse(self,
              params: ParamsTypeOpt = None,
              method: Union[str, Callable] = None,
              name: Optional[Union[str, bool]] = None,
              error_name: Optional[str] = None) -> Dict[ZfitIndependentParameter, Dict]:
        """Calculate for `params` the symmetric error using the Hessian/covariance matrix.

        Args:
            params: The parameters to calculate the
                Hessian symmetric error. If None, use all parameters.
            method: the method to calculate the covariance matrix. Can be
                {'minuit_hesse', 'hesse_np', 'approx'} or a callable.
            name: The name for the error in the dictionary. This will be added to
                the information collected in params under ``params[p][name]`` where
                p is a Parameter. If the name is `False`, it won't be added and only
                returned.
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
        # Deprecated name
        if error_name is not None:
            name = error_name
        if name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `name` or use a string as `method`")
            name = method

        with self._input_check_reset_params(params) as params:
            uncached_params = self._get_uncached_params(params=params, method_name=name)
            if uncached_params:
                error_dict = self._hesse(params=uncached_params, method=method)
                if name:
                    self._cache_errors(error_name=name, errors=error_dict)
            else:
                error_dict = {}

        error_dict.update({p: self.params[p][name] for p in params if p not in uncached_params})
        return {p: error_dict[p] for p in params}

    def _cache_errors(self, error_name, errors):
        for param, error in errors.items():
            self.params[param][error_name] = error

    def _hesse(self, params, method):
        covariance_dict = self.covariance(params, method, as_dict=True)
        return dict((p, {"error": covariance_dict[(p, p)] ** 0.5 if covariance_dict[(p, p)] is not None else None})
                    for p in params)

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
        return self.errors(params=params, method=method, name=error_name, sigma=sigma)[0]

    @deprecated_args(None, "Use name instead.", 'error_name')
    @deprecated_args(None, "Use cl for confidence level instead.", 'sigma')
    def errors(self, params: ParamsTypeOpt = None, method: Union[str, Callable] = None, name: str = None,
               error_name: str = None,
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
                name: The name for the error in the dictionary.
                error_name: The name for the error in the dictionary.


            Returns:
                A `OrderedDict` containing as keys the parameter and as value a `dict` which
                    contains (next to probably more things) two keys 'lower' and 'upper',
                    holding the calculated errors.
                    Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        """
        # Deprecated name
        if error_name is not None:
            name = error_name

        if sigma is not None:
            if cl is not None:
                raise ValueError("Cannot define sigma and cl, use cl only.")
            else:
                cl = scipy.stats.chi2(1).cdf(sigma)

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
        if name is None:
            if not isinstance(method, str):
                raise ValueError("Need to specify `error_name` or use a string as `method`")
            name = method

        if method == 'zfit_error':
            warnings.warn("'zfit_error' is still experimental and may fails.", ExperimentalFeatureWarning)

        params = self._input_check_params(params)

        with self._input_check_reset_params(self.params.keys()):
            # TODO: cache with cl!
            uncached_params = self._get_uncached_params(params=params, method_name=name)

            new_result = None

            if uncached_params:
                error_dict, new_result = self._error(params=uncached_params, method=method, cl=cl)
                if new_result is None:
                    self._cache_errors(error_name=name, errors=error_dict)
                else:
                    msg = "Invalid, a new minimum was found."
                    self._cache_errors(error_name=name, errors={p: msg for p in params})
                    self._valid = False
                    new_result._cache_errors(error_name=name, errors=error_dict)
        all_errors = OrderedDict((p, self.params[p][name]) for p in params)

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
        covariance = {k: self._covariance_dict[method].get(k) for k in itertools.product(params, params)}

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


class ListWithKeys(collections.UserList):
    __slots__ = ('_initdict',)

    def __init__(self, initdict) -> None:
        super().__init__(initlist=initdict.values())
        self._initdict = initdict

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            return self._initdict[item]
        return super().__getitem__(item)

    def keys(self):
        return self._initdict.keys()


class ValuesHolder(NameToParamGetitem, ListWithKeys):
    __slots__ = ()


class ParamHolder(NameToParamGetitem, collections.UserDict):

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

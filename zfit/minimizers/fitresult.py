#  Copyright (c) 2024 zfit

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from ..util.exception import BreakingAPIChangeError

if TYPE_CHECKING:
    import zfit

    from .evaluation import LossEval

import collections
import contextlib
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping

import colored
import iminuit
import numpy as np
import scipy.optimize
import scipy.stats
from colorama import Style, init
from ordered_set import OrderedSet
from scipy.optimize import LbfgsInvHessProduct
from tabulate import tabulate

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):  # for type checking
        import ipyopt

from ..core.interfaces import (
    ZfitData,
    ZfitIndependentParameter,
    ZfitLoss,
    ZfitParameter,
)
from ..core.parameter import set_values
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args
from ..util.warnings import ExperimentalFeatureWarning
from ..util.ztyping import ParamsTypeOpt
from ..z import numpy as znp
from .errors import (
    compute_errors,
    covariance_with_weights,
    dict_to_matrix,
    matrix_to_dict,
)
from .interface import ZfitMinimizer, ZfitResult
from .termination import ConvergenceCriterion

init(autoreset=True)


class Approximations:
    def __init__(
        self,
        params: list[ZfitParameter] | tuple[ZfitParameter],
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        inv_hessian: np.ndarray | None = None,
    ) -> None:
        """Holds different approximations after the minimisation and/or calculates them.

        Args:
            params: List of parameters the approximations (gradient, hessian, ...) were calculated with.
            gradient: Gradient
            hessian: Hessian Matrix
            inv_hessian: Inverse of the Hessian Matrix
        """
        if isinstance(params, dict):
            params = list(params)
        if not isinstance(params, (list, tuple)):
            msg = f"params has to be a list or tuple, not {type(params)}"
            raise TypeError(msg)
        if gradient is not None:
            if gradient.ndim > 1:
                msg = f"Gradient has to be a 1D array, not {gradient.shape}"
                raise ValueError(msg)
            if gradient.shape[0] != len(params):
                msg = f"Gradient has to have the same length as params, {gradient.shape[0]} != {len(params)}"
                raise ValueError(msg)

        if hessian is not None:
            if hessian.ndim != 2:
                msg = f"Hessian has to be a 2D array, not {hessian.shape}"
                raise ValueError(msg)
            if hessian.shape[0] != len(params) or hessian.shape[1] != len(params):
                msg = f"Hessian has to have the same length as params, {hessian.shape} != {len(params)}"
                raise ValueError(msg)

        if inv_hessian is not None:
            if inv_hessian.ndim != 2:
                msg = f"Inverse Hessian has to be a 2D array, not {inv_hessian.shape}"
                raise ValueError(msg)
            if inv_hessian.shape[0] != len(params) or inv_hessian.shape[1] != len(params):
                msg = f"Inverse Hessian has to have the same length as params, {inv_hessian.shape} != {len(params)}"
                raise ValueError(msg)

        self._params = params
        self._gradient = gradient
        self._hessian = hessian
        self._inv_hessian = inv_hessian
        super().__init__()

    @property
    def params(self):
        return self._params

    def gradient(self, params: ZfitParameter | Iterable[ZfitParameter] | None = None) -> np.ndarray | None:
        """Return an approximation of the gradient _if available_.

        Args:
            params: Parameters to which the gradients should be returned

        Returns:
            Array with gradients or ``None``
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

    def hessian(self, invert: bool = True) -> np.ndarray | None:
        """Return an approximation of the hessian _if available_.

        Args:
            invert: If a _hessian approximation_ is not available but an inverse hessian is, invert the latter to
                obtain the hessian approximation.

        Returns:
            Array with hessian matrix or ``None``
        """
        hess = self._hessian
        if hess is None and invert:
            inv_hess = self._inv_hessian
            if inv_hess is not None:
                hess = np.linalg.inv(inv_hess)
                self._hessian = hess
        return hess

    def inv_hessian(self, invert: bool = True) -> None | np.ndarray:
        """Return an approximation of the inverse hessian _if available_.

        Args:
            invert: If an _inverse hessian approximation_ is not available but a hessian is, invert the latter to
                obtain the inverse hessian approximation.

        Returns:
            Array with the inverse of the hessian matrix or ``None``
        """
        inv_hess = self._inv_hessian
        if inv_hess is None and invert:
            hess = self._hessian
            if hess is not None:
                inv_hess = np.linalg.inv(hess)
                self._inv_hessian = inv_hess
        return inv_hess

    def freeze(self):
        self._params = [p.name for p in self.params]


def _minos_minuit(result, params, cl=None):
    minuit_minimizer = result._create_minuit_instance()

    try:
        minuit_minimizer.minos(*(p.name for p in params), cl=cl)
        # Minuit seems very ustable on this and the call can fail after a few trials
    except RuntimeError as error:
        if "Function minimum is not valid." not in error.args[0]:
            raise
        minuit_minimizer.reset()
        minuit_minimizer.minos(*(p.name for p in params), cl=cl)

    merror_result = minuit_minimizer.merrors  # returns every var
    attrs = [
        "lower",
        "upper",
        "is_valid",
        "upper_valid",
        "lower_valid",
        "at_lower_limit",
        "at_upper_limit",
        "nfcn",
    ]
    errors = {}
    for p in params:
        error_res = merror_result[p.name]
        errors[p] = {attr: getattr(error_res, attr) for attr in attrs}
        errors[p]["original"] = error_res
    new_result = None
    return errors, new_result


def _covariance_minuit(result, params):
    minuit = result._create_minuit_instance()

    _ = minuit.hesse()  # make sure to have an accurate covariance

    covariance = minuit.covariance

    covariance_dict = {}
    if covariance is None:
        warnings.warn(
            "minuit failed to calculate the covariance matrix or similar when calling `hesse`."
            "Try to use `hesse_np` as the method instead and try again."
            "This is unexpected and may has to do with iminuitV2. Either way, please fill an issue if"
            " this is not expected to fail for you.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        for p1 in params:
            for p2 in params:
                key = (p1, p2)
                covariance_dict[key] = covariance[tuple(k.name for k in key)]

    return covariance_dict


def _covariance_np(result, params):
    if any(isinstance(data, ZfitData) and data.weights is not None for data in result.loss.data):
        warnings.warn(
            "The computation of the covariance matrix with weights is still experimental.",
            ExperimentalFeatureWarning,
            stacklevel=3,
        )

    hessian = result.loss.hessian(params)
    covariance = znp.linalg.inv(hessian)
    return matrix_to_dict(params, covariance)


def _covariance_approx(result, params):
    if any(isinstance(data, ZfitData) and data.weights is not None for data in result.loss.data):
        warnings.warn(
            "Approximate covariance/hesse estimation with weights is not supported, returning None",
            RuntimeWarning,
            stacklevel=3,
        )

    inv_hessian = result.approx.inv_hessian(invert=True)
    if inv_hessian is None:
        return {}

    params_approx = list(result.params)
    param_indices = [params_approx.index(p) for p in params]
    return {
        (p1, p2): inv_hessian[(p1_index, p2_index)]
        for (p1, p1_index), (p2, p2_index) in itertools.product(zip(params, param_indices), zip(params, param_indices))
    }


class ParamToNameGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            item = item.name
        return super().__getitem__(item)


class NameToParamGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            item = item.name
        for param in self.keys():
            name = param.name if isinstance(param, ZfitParameter) else param
            if name == item:
                item = param
                break
        return super().__getitem__(item)  # raises key error if not there, which is good

    def __contains__(self, item):
        try:
            self[item]
        except KeyError:
            return False
        except Exception as error:
            msg = "Unknown exception occurred! This should not happen."
            raise RuntimeError(msg) from error
        else:
            return True


class FitResult(ZfitResult):
    _default_hesse = "minuit_hesse"
    _hesse_methods: typing.ClassVar = {
        "minuit_hesse": _covariance_minuit,
        "hesse_np": _covariance_np,
        "approx": _covariance_approx,
    }
    _default_error = "minuit_minos"
    _error_methods: typing.ClassVar = {
        "minuit_minos": _minos_minuit,
        "zfit_error": compute_errors,
        "zfit_errors": compute_errors,
    }

    def __init__(
        self,
        loss: ZfitLoss,
        params: dict[ZfitParameter, float],
        minimizer: ZfitMinimizer,
        valid: bool,
        edm: float,
        fminopt: float,
        criterion: ConvergenceCriterion | None,
        status: int | None = None,
        converged: bool | None = None,
        message: str | None = None,
        info: Mapping | None = None,
        approx: Mapping | Approximations | None = None,
        niter: int | None = None,
        evaluator: LossEval = None,
    ) -> None:
        """Create a ``FitResult`` from a minimization. Store parameter values, minimization infos and calculate errors.

        Any errors calculated are saved under ``self.params`` dictionary with::

            {parameter: {error_name1: {'low': value, 'high': value or similar}}

        The ``FitResult can be used to temporarily update the parameter values to the values found by the minimization

        .. code-block::

            with result:
                # do something with the new values
                ...

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            info: |@doc:result.init.info| Additional information (if available)
                   such as *number of gradient function calls* or the
                   original minimizer return message.
                   This is a relatively free field and _no single field_
                   in it is guaranteed to be stable.
                   Some recommended fields:

                   - *original*: contains the original returned object
                     by the minimizer used internally.
                   - *optimizer*: the actual instance of the wrapped
                     optimizer (if available) |@docend:result.init.info|
            approx: |@doc:result.init.approx| Collection of approximations found during
                   the minimization process such as gradient and hessian. |@docend:result.init.approx|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
        """
        super().__init__()

        if status is None:
            status = 0 if valid else -999
        if converged is None and valid:
            converged = True
        if message is None:
            message = "" if valid else "Invalid, unknown reason (not specified)"

        approx = self._input_convert_approx(approx, evaluator, info, params)

        if evaluator is not None:
            niter = evaluator.niter if niter is None else niter

        info = {"n_eval": niter} if info is None else info
        param_at_limit = any(param.at_limit for param in params)
        if param_at_limit:
            valid = False
            if message:
                message += " AND "
            message += "parameter(s) at their limit."

        self._cache_minuit = None  # in case used in errors

        self._evaluator = evaluator  # keep private for now
        self._niter = niter  # keep private for now
        self._approx = approx
        self._status = status
        self._message = "" if message is None else message
        self._converged = converged
        self._params = self._input_convert_params(params)
        self._values = ValuesHolder(params)
        self._params_at_limit = param_at_limit
        self._edm = edm
        self._criterion = criterion
        self._fminopt = fminopt
        self._info = info
        self._loss = loss
        self._minimizer = minimizer
        self._valid = valid
        self._covariance_dict = {}
        self._tmp_old_param_values = []
        try:
            fminfull = loss.value(full=True)
        except Exception as error:
            warnings.warn(
                f"Could not calculate fminfull due to {error}. Setting to 0. This is a new feature and is caught to not break backwards compatibility.",
                RuntimeWarning,
                stacklevel=3,
            )
            fminfull = 0
        self._fmin = float(fminfull)

    def _input_convert_approx(self, approx, evaluator, info, params):
        """Convert approx (if a Mapping) to an `Approximation` using the information provided.

        Args:
            approx: |@doc:result.init.approx| Collection of approximations found during
                   the minimization process such as gradient and hessian. |@docend:result.init.approx|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
            info: |@doc:result.init.info| Additional information (if available)
                   such as *number of gradient function calls* or the
                   original minimizer return message.
                   This is a relatively free field and _no single field_
                   in it is guaranteed to be stable.
                   Some recommended fields:

                   - *original*: contains the original returned object
                     by the minimizer used internally.
                   - *optimizer*: the actual instance of the wrapped
                     optimizer (if available) |@docend:result.init.info|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|

        Returns:
            The created approximation.
        """
        approx = {} if approx is None else approx
        if isinstance(approx, collections.abc.Mapping):
            if "params" not in approx:
                approx["params"] = params

            if info:
                if "gradient" not in approx:
                    approx["gradient"] = info.get("grad", info.get("gradient"))
                if "hessian" not in approx:
                    approx["hessian"] = info.get("hess", info.get("hesse", info.get("hessian")))
                if "inv_hessian" not in approx:
                    approx["inv_hessian"] = info.get("inv_hess", info.get("inv_hesse", info.get("inv_hessian")))
            if evaluator is not None:
                if "gradient" not in approx:
                    approx["gradient"] = evaluator.last_gradient
                if "hessian" not in approx:
                    approx["hessian"] = evaluator.last_hessian

            approx = Approximations(**approx)
        return approx

    def _input_convert_params(self, params):
        return ParamHolder((p, {"value": v}) for p, v in params.items())

    def _check_get_uncached_params(self, params, method_name, cl):
        uncached = []
        for p in params:
            errordict = self.params[p].get(method_name)
            # cl is < 1 and gets very close. The closer, the more it matters -> scale tolerance by it
            if errordict is not None and not math.isclose(errordict["cl"], cl, abs_tol=3e-3 * (1 - cl)):
                msg = (
                    f"Error with name {method_name} already exists in {self!r} with a different"
                    f" convidence level of {errordict['cl']} instead of the requested {cl}."
                    f" Use a different name."
                )
                raise NameError(
                    msg,
                )
            uncached.append(p)
        return uncached

    def _create_minuit_instance(self):
        minuit = self._cache_minuit
        from zfit.minimizers.minimizer_minuit import Minuit

        if minuit is None:
            if isinstance(self.minimizer, Minuit):
                minuit = self.minimizer._minuit_minimizer
            else:
                minimizer = Minuit(tol=self.minimizer.tol, verbosity=0, name="ZFIT_TMP_UNCERTAINITIES")
                minuit, _, _ = minimizer._make_minuit(loss=self.loss, params=self.params, init=self)
            self._cache_minuit = minuit
        return minuit

    @classmethod
    def from_ipopt(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        problem: ipyopt.Problem,
        minimizer: zfit.minimize.Ipyopt,
        valid: bool,
        values: np.ndarray,
        message: str | None,
        converged: bool | None,
        edm: zfit.minimizers.termination.CriterionNotAvailable | float,
        niter: int | None,
        fminopt: float | None,
        status: int | None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion,
        evaluator: zfit.minimizers.evaluation.LossEval | None,
    ) -> FitResult:
        """Create a ``FitResult`` from an ipopt minimization.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            problem: |@doc:result.init.problem||@docend:result.init.problem|
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|

        Returns:
            ``zfit.minimize.FitResult``:
        """
        info = {"problem": problem}
        params = dict(zip(params, values))
        valid = valid if converged is None else valid and converged
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached
        return cls(
            params=params,
            loss=loss,
            fminopt=fminopt,
            edm=edm,
            message=message,
            criterion=criterion,
            info=info,
            valid=valid,
            converged=converged,
            niter=niter,
            status=status,
            minimizer=minimizer,
            evaluator=evaluator,
        )

    @classmethod
    def from_minuit(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        minuit: iminuit.Minuit,
        minimizer: ZfitMinimizer | iminuit.Minuit,
        valid: bool | None,
        values: np.ndarray | None = None,
        message: str | None = None,
        converged: bool | None = None,
        edm: None | (zfit.minimizers.termination.CriterionNotAvailable | float) = None,
        niter: int | None = None,
        fminopt: float | None = None,
        status: int | None = None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion | None = None,
        evaluator: zfit.minimizers.evaluation.LossEval | None = None,
    ) -> FitResult:
        """Create a `FitResult` from a :py:class:`~iminuit.util.MigradResult` returned by
        :py:meth:`iminuit.Minuit.migrad` and a iminuit :py:class:`~iminuit.Minuit` instance with the corresponding zfit
        objects.

        Args:
            loss: zfit Loss that was minimized.
            params: Iterable of the zfit parameters that were floating during the minimization.
            minuit: Return value of the iminuit migrad command, the instance of :class:`iminuit.Minuit`
            minimizer: Instance of the zfit Minuit minimizer that was used to minimize the loss.
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|


        Returns:
            ``zfit.minimize.FitResult``: A `FitResult` as if zfit Minuit was used.
        """
        from .minimizer_minuit import Minuit
        from .termination import EDM

        if not isinstance(minimizer, Minuit):
            if isinstance(minimizer, iminuit.Minuit):
                minimizer_new = Minuit()
                minimizer_new._minuit_minimizer = minimizer
                minimizer = minimizer_new
            else:
                msg = f"Minimizer {minimizer} not supported. Use `Minuit` from zfit or from iminuit."
                raise ValueError(msg)

        params_result = list(minuit.params)

        fmin_object = minuit.fmin
        minuit_converged = not fmin_object.is_above_max_edm
        converged = minuit_converged if converged is None else (converged and minuit_converged)
        niter = fmin_object.nfcn if niter is None else niter
        info = {
            "n_eval": niter,
            # 'grad': result['jac'],
            # 'message': result['message'],
            "minuit": minuit,
            "original": fmin_object,
        }
        if fmin_object.has_covariance:
            info["inv_hessian"] = np.array(minuit.covariance)

        edm = fmin_object.edm if edm is None else edm
        if criterion is None:
            criterion = EDM(tol=minimizer.tol, loss=loss, params=params)
            criterion.last_value = edm
        fminopt = fmin_object.fval if fminopt is None else fminopt
        minuit_valid = fmin_object.is_valid
        valid = minuit_valid if valid is None else minuit_valid and valid
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached
        if values is None:
            values = tuple(res.value for res in params_result)
        params = dict(zip(params, values))
        return cls(
            params=params,
            edm=edm,
            fminopt=fminopt,
            info=info,
            loss=loss,
            niter=niter,
            converged=converged,
            status=status,
            message=message,
            valid=valid,
            criterion=criterion,
            minimizer=minimizer,
            evaluator=evaluator,
        )

    @classmethod
    def from_scipy(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        result: scipy.optimize.OptimizeResult,
        minimizer: ZfitMinimizer,
        message: str | None,
        valid: bool,
        criterion: ConvergenceCriterion,
        edm: float | None = None,
        niter: int | None = None,
        evaluator: zfit.minimize.LossEval | None = None,
    ) -> FitResult:
        """Create a ``FitResult from a SciPy `~scipy.optimize.OptimizeResult`.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            result: Result of the SciPy optimization.
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|

        Returns:
            `zfit.minimize.FitResult`:
        """
        result_values = result["x"]
        if niter is None:
            niter = result.get("nit", 0)

        converged = result.get("success", valid)
        status = result["status"]

        if message is None and (not converged or not valid):
            message = result.get("message")
        grad = result.get("grad")
        info = {
            "n_eval": result["nfev"],
            "n_iter": niter,
            "niter": niter,
            "grad": result.get("jac") if grad is None else grad,
            "message": message,
            "evaluator": evaluator,
            "original": result,
        }
        approx = {
            "params": params,
            "gradient": info.get("grad"),
        }
        if info.get("niter", 0) > 25:  # unreliable if too few iterations, fails for EDM
            inv_hesse = result.get("hess_inv")
            if isinstance(inv_hesse, LbfgsInvHessProduct):
                inv_hesse = inv_hesse.todense()
            hesse = info.get("hesse")
            info["inv_hesse"] = inv_hesse
            info["hesse"] = hesse
            approx["hessian"] = hesse
            approx["inv_hessian"] = inv_hesse

        fminopt = result["fun"]
        params = dict(zip(params, result_values))
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached

        return cls(
            params=params,
            edm=edm,
            fminopt=fminopt,
            info=info,
            approx=approx,
            converged=converged,
            status=status,
            message=message,
            valid=valid,
            niter=niter,
            loss=loss,
            minimizer=minimizer,
            criterion=criterion,
            evaluator=evaluator,
        )

    @classmethod
    def from_nlopt(
        cls,
        loss: ZfitLoss,
        opt,
        params: Iterable[ZfitParameter],
        minimizer: ZfitMinimizer | iminuit.Minuit,
        valid: bool | None,
        values: np.ndarray | None = None,
        message: str | None = None,
        converged: bool | None = None,
        edm: None | (zfit.minimizers.termination.CriterionNotAvailable | float) = None,
        niter: int | None = None,
        fminopt: float | None = None,
        status: int | None = None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion | None = None,
        evaluator: zfit.minimizers.evaluation.LossEval | None = None,
        inv_hessian: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
    ) -> FitResult:
        """Create a ``FitResult`` from an NLopt optimizer.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            opt: Optimizer instance of NLopt
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
            inv_hessian: The (approximated) inverse hessian matrix.
            hessian: The (approximated) hessian matrix.

        Returns:
            zfit.minimizers.fitresult.FitResult:
        """
        converged = converged if converged is None else bool(converged)
        param_dict = dict(zip(params, values))
        if fminopt is None:
            fminopt = opt.last_optimum_value()
        status_nlopt = opt.last_optimize_result()
        if status is None:
            status = status_nlopt
        niter = opt.get_numevals() if niter is None else niter
        converged = 1 <= status_nlopt <= 4 and converged is not False

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
        message_nlopt = messages[status_nlopt]
        info = {
            "n_eval": niter,
            "niter": niter,
            "message": message_nlopt,
            "original": status,
            "evaluator": evaluator,
            "status": status,
        }
        if message is None:
            message = message_nlopt

        valid = valid and converged
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached

        approx = {}
        if inv_hessian is None and hessian is None and evaluator is not None:
            hessian = evaluator.last_hessian
            # if hessian is not None:  # TODO: remove?
            #     inv_hessian = np.linalg.inv(hessian)

        if inv_hessian is not None:
            info["inv_hesse"] = inv_hessian
            approx["inv_hessian"] = inv_hessian

        return cls(
            params=param_dict,
            edm=edm,
            fminopt=fminopt,
            status=status,
            converged=converged,
            info=info,
            niter=niter,
            valid=valid,
            loss=loss,
            minimizer=minimizer,
            criterion=criterion,
            message=message,
            evaluator=evaluator,
        )

    @property
    def approx(self) -> Approximations:
        return self._approx

    @property
    def params(
        self,
    ) -> Mapping[ZfitIndependentParameter, Mapping[str, Mapping[str, object]]]:
        return self._params

    @property
    def values(self) -> Mapping[str | ZfitParameter, float]:
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
    def fminopt(self) -> float:
        """Function value with possible optimizations at the minimum, namely constant subtraction.

        Returns:
            Numeric
        """
        return self._fminopt

    @property
    def fmin(self) -> float:
        """Function value, fully evaluated, at the minimum.

        Returns:
            Numeric
        """
        return self._fmin

    @property
    @deprecated(
        None,
        "Use `fmin` instead which now returns the full minimum value. This will be removed in the future.",
    )
    def fminfull(self) -> float:
        """Function value, fully evaluated, at the minimum.

        Returns:
            Numeric
        """
        return self.fmin

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
        old_values = np.asarray(params)
        try:
            yield params
        except Exception:
            warnings.warn(
                "Exception occurred, parameter values are not reset and in an arbitrary, last"
                " used state. If this happens during normal operation, make sure you reset the values.",
                RuntimeWarning,
                stacklevel=3,
            )
            raise
        set_values(params=params, values=old_values, allow_partial=True)  # TODO: or set?

    def _input_check_params(self, params):
        return convert_to_container(params) if params is not None else list(self.params.keys())

    @deprecated_args(None, "Use `name` instead", "error_name")
    def hesse(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        cl: float | None = None,
        name: str | bool | None = None,
        # DEPRECATED
        error_name: str | None = None,
    ) -> dict[ZfitIndependentParameter, dict]:
        r"""Calculate for `params` the symmetric error using the Hessian/covariance matrix.

        This method estimates the covariance matrix using the inverse of the Hessian matrix. The assumption is
        that the loss profile - usually a likelihood or a :math:`\chi^2` - is hyperbolic. This is usually the case for
        fits with many observations, i.e. it is exact in the asymptotic limit. If the loss profile is not hyperbolic,
        another method, "zfit_error" or "minuit_minos" should be used.

        **Weights**
        Weighted likelihoods are a special class of likelihoods as they are not an actual likelihood. However, the
        minimum is still valid, however the profile is not a proper likelihood. Therefore, corrections
        will be automatically applied to the Hessian uncertainty estimation in order to correct for the effects
        in the weights. The corrections used are "asymptotically correct" and are described in
        `Parameter uncertainties in weighted unbinned maximum likelihood fits`<https://doi.org/10.1140/epjc/s10052-022-10254-8>`
        by Christoph Langenbruch.
        Since this method uses the jacobian matrix, it takes significantly longer to calculate than witout weights.


        Args:
            params: The parameters to calculate the
                Hessian symmetric error. If None, use all parameters.
            method: the method to calculate the covariance matrix. Can be
                {'minuit_hesse', 'hesse_np', 'approx'} or a callable.
            cl: Confidence level for the error. If None, use the default value of 0.68.
            name: The name for the error in the dictionary. This will be added to
                the information collected in params under ``params[p][name]`` where
                p is a Parameter. If the name is `False`, it won't be added and only
                returned. Defaulst to `'hesse'`.

        Returns:
            Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'error': sym_error}.

                So given param_a (from zfit.Parameter(.))
                `error_a = result.hesse(params=param_a)[param_a]['error']`
                error_a is the hessian error.
        """
        # for compatibility with `errors`
        cl = 0.68268949 if cl is None else cl  # scipy.stats.chi2(1).cdf(1)
        if cl >= 1:
            msg = f"cl is the confidence limit and has to be < 1, not {cl}"
            raise ValueError(msg)

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

        name_warning_triggered = False
        if name is None:
            if not isinstance(method, str):
                msg = "Need to specify `name` or use a string as `method`"
                raise ValueError(msg)
            name = "hesse"

        with self._input_check_reset_params(params) as checkedparams:
            uncached_params = self._check_get_uncached_params(params=checkedparams, method_name=name, cl=cl)
            if uncached_params:
                error_dict = self._hesse(params=uncached_params, method=method, cl=cl)
                if any(val["error"] is None for val in error_dict.values()):
                    return {}
                for p in error_dict:
                    error_dict[p]["cl"] = cl
                if name:
                    self._cache_errors(name=name, errors=error_dict)
            else:
                error_dict = {}

        error_dict.update({p: self.params[p][name] for p in checkedparams if p not in uncached_params})
        if name_warning_triggered:
            error_dict.update({p: self.params[p][method] for p in checkedparams if p not in uncached_params})
        return {p: error_dict[p] for p in checkedparams}

    def _cache_errors(self, name, errors):
        for param, error in errors.items():
            self.params[param][name] = error

    def _hesse(self, params, method, cl):
        pseudo_sigma = scipy.stats.chi2(1).ppf(cl) ** 0.5

        covariance_dict = self.covariance(params, method, as_dict=True)
        return {
            p: {
                "error": (
                    float(covariance_dict[(p, p)]) ** 0.5 * pseudo_sigma
                    if covariance_dict[(p, p)] is not None
                    else None
                )
            }
            for p in params
        }

    def error(
        self,
        params: ParamsTypeOpt = None,  # noqa: ARG002
        method: str | Callable | None = None,  # noqa: ARG002
        error_name: str | None = None,  # noqa: ARG002
        sigma: float = 1.0,  # noqa: ARG002
    ) -> dict:
        r""".. deprecated:: unknown Use :func:`errors` instead.

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
            A `dict` containing as keys the parameter names and as value a `dict` which
                contains (next to probably more things) two keys 'lower' and 'upper',
                holding the calculated errors.
                Example: result['par1']['upper'] -> the asymmetric upper error of 'par1'
        """
        msg = "Use `errors` instead of `error`."
        raise BreakingAPIChangeError(msg)

    @deprecated_args(None, "Use name instead.", "error_name")
    def errors(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        name: str | None = None,
        cl: float | None = None,
        *,
        sigma=None,
        error_name: str | None = None,
    ) -> tuple[dict, None | FitResult]:
        r"""Calculate and set for `params` the asymmetric error using the set error method.

        Args:
            params: The parameters or their names to calculate the
                 errors. If `params` is `None`, use all *floating* parameters.
            method: The method to use to calculate the errors. Valid choices are
                {'minuit_minos', 'zfit_errors'} or a Callable.
            cl: Uncertainties are calculated with respect to the confidence level cl. The default is 68.3%.
                For example, the negative log-likelihood (without the factor of 2) has a correspondents
                of :math:`\Delta` NLL of 1 corresponds to 1 std deviation.
            sigma: Errors are calculated with respect to `sigma` std deviations. The definition
                of 1 sigma depends on the loss function and is defined there.
            name: The name for the error in the dictionary. Defaults to `errors`


        Returns:
            A `dict` containing as keys the parameter and as value a `dict` which
                contains (next to often more things) two keys 'lower' and 'upper',
                holding the calculated errors. Furthermore, it has `cl` to indicate the convidence level
                the uncertainty was calculated with.
                Example: result[par1]['upper'] -> the asymmetric upper error of 'par1'
        """
        # Deprecated name
        if error_name is not None:
            name = error_name

        if sigma is not None:
            if cl is not None:
                msg = "Cannot define sigma and cl, use only one."
                raise ValueError(msg)
            cl = scipy.stats.chi2(1).cdf(sigma)

        if cl is None:
            cl = 0.68268949  # scipy.stats.chi2(1).cdf(1)

        if method is None:
            method = self._default_error
        if name is None:
            if not isinstance(method, str):
                msg = "Need to specify `name` or use a string as `method`"
                raise ValueError(msg)
            name = "errors"

        if method == "zfit_error":
            warnings.warn(
                "'zfit_error' is still somewhat new. If it fails, please report it here:"
                " https://github.com/zfit/zfit/issues/new?assignees=&labels=bug&template"
                "=bug_report.md&title=zfit%20error%20fails.",
                ExperimentalFeatureWarning,
                stacklevel=2,
            )

        params = self._input_check_params(params)

        with self._input_check_reset_params(self.params.keys()):
            uncached_params = self._check_get_uncached_params(params=params, method_name=name, cl=cl)

            new_result = None

            if uncached_params:
                error_dict, new_result = self._error(params=uncached_params, method=method, cl=cl)
                for p in error_dict:
                    error_dict[p]["cl"] = cl
                self._cache_errors(name=name, errors=error_dict)

                if new_result is not None:
                    msg = "Invalid, a new minimum was found."
                    self._cache_errors(name=name, errors={p: msg for p in params})
                    self._valid = False
                    self._message = msg
                    new_result._cache_errors(name=name, errors=error_dict)
        all_errors = {p: self.params[p][name] for p in params}

        return all_errors, new_result

    def _error(self, params, method, cl):
        if not callable(method):
            try:
                method = self._error_methods[method]
            except KeyError:
                msg = f"The following method is not a valid, implemented method: {method}. Use one of {self._error_methods.keys()}"
                raise KeyError(msg) from None
        return method(result=self, params=params, cl=cl)

    def covariance(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        as_dict: bool = False,
    ):
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
            method = self._default_hesse

        if method not in self._covariance_dict:
            with self._input_check_reset_params(params) as checkedparams:
                self._covariance_dict[method] = self._covariance(method=method)

        else:
            checkedparams = self._input_check_params(params)
        covariance = {k: self._covariance_dict[method].get(k) for k in itertools.product(checkedparams, checkedparams)}

        if as_dict:
            return covariance
        return dict_to_matrix(checkedparams, covariance)

    def _covariance(self, method):
        if not callable(method):
            if method not in self._hesse_methods:
                msg = f"The following method is not a valid, implemented method: {method}. Use one of {self._hesse_methods.keys()}"
                raise KeyError(msg)
            method = self._hesse_methods[method]
        params = list(self.params.keys())

        if any(isinstance(data, ZfitData) and data.weights is not None for data in self.loss.data):
            return covariance_with_weights(method=method, result=self, params=params)

        return method(result=self, params=params)

    def correlation(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        as_dict: bool = False,
    ):
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
        return correlation

    def freeze(self):
        """Freeze the result to make it pickleable and convert all TensorFlow elements to names (parameters) or arrays.

        After this, no more uncertainties or covariances can be calculated. The already calculated ones remain however.

        Parameters can be accessed by their string name.
        """
        self._loss = self.loss.name
        self._minimizer = self.minimizer.name
        self._criterion = self.criterion.name
        self._evaluator = None
        self.approx.freeze()
        self._covariance_dict = {
            k: {(p[0].name, p[1].name): v for p, v in d.items()} for k, d in self._covariance_dict.items()
        }
        self._values = ValuesHolder({p.name: self.values[p] for p in self.params})
        self._params = ParamHolder({k.name: v for k, v in self.params.items()})

        if "minuit" in self.info:
            self.info["minuit"] = "Minuit_frozen"
        if "problem" in self.info:
            try:
                import ipyopt
            except ImportError:
                pass
            else:
                if isinstance(self.info["problem"], ipyopt.Problem):
                    self.info["problem"] = "ipyopt_frozen"

        if "evaluator" in self.info:
            self.info["evaluator"] = "evaluator_frozen"
        self._cache_minuit = None

    def __str__(self):
        string = Style.BRIGHT + "FitResult" + Style.NORMAL + f" of\n{self.loss} \nwith\n{self.minimizer}\n\n"
        string += tabulate(
            [
                [
                    color_on_bool(self.valid),
                    color_on_bool(self.converged, on_true=False),
                    color_on_bool(self.params_at_limit, on_true=colored.bg(9), on_false=False),
                    format_value(self.edm, highprec=False),
                    f"          {self._fmin:.2f} | {format_value(self.fminopt)}",
                ]
            ],
            [
                "valid",
                "converged",
                "param at limit",
                "edm",
                "approx. fmin (full | opt.)",
            ],
            tablefmt="fancy_grid",
            disable_numparse=True,
            colalign=["center", "center", "center", "center", "right"],
        )
        string += "\n\n" + Style.BRIGHT + "Parameters\n" + Style.NORMAL
        string += str(self.params)
        return string

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(self.__repr__())
            return
        p.text(self.__str__())

    def __enter__(self):
        self._tmp_old_param_values.append(znp.asarray(tuple(self.params.keys())))
        self.update_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        old_vals = self._tmp_old_param_values.pop()
        set_values(tuple(self.params.keys()), old_vals)

    def update_params(self) -> typing.Self:
        """Update the parameters of the result to the current values.

        Usually to be used chained with a ``minimize`` call to update the parameters to the current values.

        (currently, the parameters are updated to the values of the result, this might change in the future. To enable
        this expected behavior, use ``zfit.run.experimental_disable_param_update()``.

        .. code-block:: python

            result = minimizer.minimize(loss).update_params()  # update the parameters to the current values
        """
        set_values(self)
        return self


def covariance_to_correlation(covariance):
    diag = np.diag(1 / np.diag(covariance) ** 0.5)
    return np.matmul(diag, np.matmul(covariance, diag))


def format_value(value, highprec=True):
    m_error_class = iminuit.util.MError  # if iminuit is not available (maybe in the future?), use dict instead

    if isinstance(value, dict) and "error" in value:
        value = value["error"]
        value = f"{value:> 6.2g}"
        value = f'+/-{" " * (8 - len(value))}' + value
    if isinstance(value, m_error_class) or (isinstance(value, dict) and "lower" in value and "upper" in value):
        if isinstance(value, m_error_class):
            lower = value.lower
            upper = value.upper
        else:
            lower = value["lower"]
            upper = value["upper"]
        lower_sign = f"{np.sign(lower): >+}"[0]
        upper_sign = f"{np.sign(upper): >+}"[0]
        lower, upper = f"{np.abs(lower): >6.2g}", f"{upper: >6.2g}"
        lower = lower_sign + " " * (7 - len(lower)) + lower
        upper = upper_sign + " " * (7 - len(upper)) + upper
        # lower += " t" * (11 - len(lower))
        value = lower + " " * 3 + upper

    if isinstance(value, float):
        value = f"{value:> 6.7g}" if highprec else f"{value:> 6.2g}"
    return value


def color_on_bool(value, on_true=None, on_false=None):
    if on_true is None:
        on_true = colored.bg(10)
    if on_false is None:
        on_false = colored.bg(9)

    if not value and on_false:
        value_add = on_false
    elif value and on_true:
        value_add = on_true
    else:
        value_add = ""
    return value_add + str(value) + Style.RESET_ALL


class ListWithKeys(collections.UserList):
    __slots__ = ("_initdict",)

    def __init__(self, initdict) -> None:
        super().__init__(initlist=initdict.values())
        self._initdict = initdict

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            return self._initdict[item]
        return super().__getitem__(item)

    def keys(self):
        return self._initdict.keys()

    def values(self):
        return self._initdict.values()

    def items(self):
        return self._initdict.items()


class ValuesHolder(NameToParamGetitem, ListWithKeys):
    __slots__ = ()


class ParamHolder(NameToParamGetitem, collections.UserDict):
    def __str__(self) -> str:
        order_keys = ["value", "hesse"]
        keys = OrderedSet()
        for pdict in self.values():
            keys.update(OrderedSet(pdict))
        order_keys = OrderedSet([key for key in order_keys if key in keys])
        order_keys.update(keys)

        rows = []
        for param, pdict in self.items():
            name = param.name if isinstance(param, ZfitParameter) else param
            row = [name]
            row.extend(format_value(pdict.get(key, " ")) for key in order_keys)
            if isinstance(param, ZfitParameter):
                row.append(
                    color_on_bool(
                        bool(param.at_limit),
                        on_true=colored.bg("light_red"),
                        on_false=False,
                    )
                )
            rows.append(row)

        order_keys = ["name", *list(order_keys), "at limit"]
        order_keys[order_keys.index("value")] = "value  (rounded)"
        return tabulate(rows, order_keys, numalign="right", stralign="right", colalign=("left",))

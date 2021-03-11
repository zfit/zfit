#  Copyright (c) 2021 zfit
"""
Definition of minimizers, wrappers etc.

"""

import collections
import copy
import functools
import inspect
import math
import os
import warnings
from typing import Union, Iterable, Mapping, Callable, Tuple, Optional, Dict

import numpy as np
from ordered_set import OrderedSet

from .evaluation import LossEval
from .fitresult import FitResult
from .interface import ZfitMinimizer, ZfitResult
from .strategy import FailMinimizeNaN, ZfitStrategy, PushbackStrategy
from .termination import EDM, ConvergenceCriterion
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..core.parameter import set_values, convert_to_parameter
from ..settings import run
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import MinimizeNotImplementedError, MinimizeStepNotImplementedError, MinimizerSubclassingError, \
    InitNotImplemented, ParameterNotIndependentError, MaximumIterationReached

DefaultStrategy = PushbackStrategy


def minimize_supports(*, init: Union[bool] = False) -> Callable:
    """Decorator: Add (mandatory for some methods) on a method to control what it can handle.

    If any of the flags is set to False, it will check the arguments and, in case they match a flag
    (say if a *init* is passed while the *init* flag is set to `False`), it will
    raise a corresponding exception (in this example a `FromResultNotImplemented`) that will
    be caught by an outer function that knows how to handle things.

    Args:
        init: Specify whether the minimize method can handle a FitResult instead of a loss as a loss. There are
            three options:
            - False: This is the default and means that _no FitResult will ever come true_. The minimizer handles the
              initial parameter values himselves.
            - 'same': If 'same' is set, a `FitResult` will only come through if it was created with the *exact* same
              type as
        multiple_limits: If False, only simple limits are to be expected and no iteration is
            therefore required.
    """

    def wrapper(func):

        parameters = inspect.signature(func).parameters
        keys = list(parameters.keys())
        init_str = 'loss'
        if init is True or init_str not in keys:  # no init as parameters -> no problem
            new_func = func
        else:
            init_index = keys.index(init_str)

            @functools.wraps(func)
            def new_func(*args, **kwargs):
                self_minimizer = args[0]
                can_handle = True
                loss_is_arg = len(args) > init_index
                if loss_is_arg:
                    init_result = args[init_index]
                else:
                    init_result = kwargs[init_str]

                if isinstance(init_result, FitResult):
                    if init == 'same':
                        if not type(self_minimizer) == type(init_result.minimizer):
                            can_handle = False
                    elif not init:
                        can_handle = False
                    else:
                        raise ValueError("`init` has to be True, False or 'same'")
                if not can_handle:
                    raise InitNotImplemented
                return func(*args, **kwargs)

        new_func.__wrapped__ = minimize_supports
        return new_func

    return wrapper


_Minimizer_CHECK_HAS_SUPPORT = {}


def _Minimizer_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support: If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.

    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func:

        Returns:
            Function:
        """
        name = func.__name__
        _Minimizer_CHECK_HAS_SUPPORT[name] = has_support
        func.__wrapped__ = _Minimizer_register_check_support
        return func

    return register


class BaseMinimizer(ZfitMinimizer):

    _DEFAULTS = {
        'tol': 1e-3,
        'verbosity': 5,
        'strategy': DefaultStrategy,
        'criterion': EDM,
        'maxiter': 'auto',
    }

    def __init__(self,
                 tol: Optional[float],
                 verbosity: Optional[int],
                 minimizer_options: Optional[Dict],
                 criterion: Optional[ConvergenceCriterion],
                 strategy: Optional[ZfitStrategy],
                 maxiter: Optional[Union[str, int]],
                 name: Optional[str],
                 **kwargs) -> None:
        """Base Minimizer to minimize loss functions and return a result.

        This class acts as a base class to implement a minimizer. The method `minimize` has to be overridden.


        Args:
            tol ():
            verbosity ():
            minimizer_options ():
            criterion ():
            strategy ():
            maxiter ():
            name ():
            **kwargs ():
        """
        super().__init__(**kwargs)
        self._n_iter_per_param = 1000

        self.tol = self._DEFAULTS['tol'] if tol is None else tol
        self.verbosity = self._DEFAULTS['verbosity'] if verbosity is None else verbosity
        self.minimizer_options = {} if minimizer_options is None else minimizer_options
        self.criterion = self._DEFAULTS['criterion'] if criterion is None else criterion

        if strategy is None:
            strategy = self._DEFAULTS['strategy']()
        if not isinstance(strategy, ZfitStrategy):
            raise TypeError(f"strategy {strategy} is not an instance of ZfitStrategy.")
        self.strategy = strategy
        self.maxiter = self._DEFAULTS['maxiter'] if maxiter is None else maxiter
        self.name = repr(self.__class__)[:-2].split(".")[-1] if name is None else name

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if subclass has decorator if required
        cls._subclass_check_support(methods_to_check=_Minimizer_CHECK_HAS_SUPPORT,
                                    wrapper_not_overwritten=_Minimizer_register_check_support)

    @classmethod
    def _subclass_check_support(cls, methods_to_check, wrapper_not_overwritten):
        for method_name, has_support in methods_to_check.items():
            if not hasattr(cls, method_name):
                continue  # skip if only subclass requires it
            method = getattr(cls, method_name)
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == wrapper_not_overwritten:
                    continue  # not overwritten, fine

            # here means: overwritten
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == minimize_supports:
                    if has_support:
                        continue  # needs support, has been wrapped
                    else:
                        raise MinimizerSubclassingError("Method {} has been wrapped with minimize_supports "
                                                        "but is not allowed to. Has to handle all "
                                                        "arguments.".format(method_name))
                elif has_support:
                    raise MinimizerSubclassingError(f"Method {method_name} has been overwritten and *has to* be "
                                                    "wrapped by `@minimize_supports` decorator (don't forget () )"
                                                    "to call the decorator as it takes arguments")
                elif not has_support:
                    continue  # no support, has not been wrapped with
            else:
                if not has_support:
                    continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            raise MinimizerSubclassingError(f"Method {method_name} has not been correctly wrapped with "
                                            f"@minimize_supports ")

    def _check_convert_input(self, loss: ZfitLoss, params, init=None, only_floating=True
                             ) -> Tuple[ZfitLoss, Iterable[ZfitParameter], Union[None, FitResult]]:
        """Sanitize the input values and return all of them.



        Args:
            loss: If the loss is a callable, it will be converted to a SimpleLoss.
            params: If the parameters is an array, it will be converted to free parameters.
            init:
            only_floating:

        Returns:
            loss, params, init:

        """
        if not isinstance(loss, ZfitLoss):
            if not callable(loss):
                raise TypeError("Given Loss has to  be a ZfitLoss or a callable.")
            elif params is None:
                raise ValueError("If the loss is a callable, the params cannot be None.")

            if not isinstance(params, collections.Mapping):
                values = convert_to_container(params)
                names = [None] * len(params)
            else:
                values = list(params.values())
                names = list(params.keys())
            params = [convert_to_parameter(value=val, name=name, prefer_constant=False)
                      for val, name in zip(values, names)]

            from zfit.core.loss import SimpleLoss
            if hasattr(loss, 'errordef'):
                errordef = loss.errordef
            else:
                errordef = 0.5
            loss = SimpleLoss(func=loss, deps=params, errordef=errordef)

        to_set_param_values = {}
        if params is None:
            params = loss.get_params(only_floating=only_floating)
        else:
            if isinstance(params, collections.Mapping):
                param_values = params
                to_set_param_values = {p: val for p, val in param_values.items() if val is not None}
                try:
                    set_values(list(to_set_param_values), list(to_set_param_values.values()))
                except ParameterNotIndependentError as error:
                    not_indep_and_set = {p for p, val in param_values.items() if val is not None and not p.independent}
                    raise ParameterNotIndependentError(f"Cannot set parameter {not_indep_and_set} to a value as they"
                                                       f" are not independent. The following `param` argument was"
                                                       f" given: {params}."
                                                       f""
                                                       f"Original error"
                                                       f"--------------"
                                                       f"{error}") from error
            else:
                params = convert_to_container(params, container=OrderedSet)

            # now extract all the independent parameters
            params = list(OrderedSet.union(*(p.get_params(only_floating=only_floating) for p in params)))

        # set the parameter values from the init
        if init is not None:
            # don't set the user set
            params_to_set = OrderedSet(params).intersection(OrderedSet(init.params)) - OrderedSet(to_set_param_values)
            set_values(params_to_set, init)
        if only_floating:
            params = self._filter_floating_params(params)
        if not params:
            raise RuntimeError("No parameter for minimization given/found. Cannot minimize.")
        params = list(params)
        return loss, params, init

    @staticmethod
    def _filter_floating_params(params):
        non_floating = [param for param in params if not param.floating]
        if non_floating:  # legacy warning
            warnings.warn(f"CHANGED BEHAVIOR! Non-floating parameters {non_floating} will not be used in the "
                          f"minimization.")
        return [param for param in params if param.floating]

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    def minimize(self,
                 loss: Union[ZfitLoss, Callable],
                 params: Optional[ztyping.ParamsTypeOpt] = None,
                 init: Optional[ZfitResult] = None
                 ) -> FitResult:
        """Fully minimize the `loss` with respect to `params`, optionally using information from `init`.

        Args:
            loss: Loss to be minimized.
            params: The parameters with respect to which to
                minimize the `loss`. If `None`, the parameters will be taken from the `loss`.
            init: A result of a previous minimization that provides auxiliary information such as the starting point for
                the parameters

        Returns:
            The fit result.
        """
        if isinstance(loss, ZfitResult):
            init = loss  # make the names correct
            loss = init.loss

        loss, params, init = self._check_convert_input(loss=loss, params=params, init=init, only_floating=True)

        return self._call_minimize(loss=loss, params=params, init=init)

    def _call_minimize(self, loss, params, init):
        do_recovery = False
        prelim_result = None
        try:
            result =  self._minimize(loss=loss, params=params, init=init)
        except TypeError as error:
            if "got an unexpected keyword argument 'init'" in error.args[0]:
                warnings.warn(
                    '_minimize has to take an `init` argument. This will be mandatory in the future, please'
                    ' change the signature accordingly.', category=FutureWarning, stacklevel=2)
                result = self._call_minimize(loss=loss, params=params)
            else:
                raise
        # TODO: catch `init` not supported?
        except (FailMinimizeNaN, RuntimeError):  # iminuit raises RuntimeError if user raises Error
            do_recovery = True
            prelim_result = self.strategy.fit_result
            if prelim_result is not None:

                result = prelim_result
            else:
                raise
        except MaximumIterationReached:
            do_recovery = True
            # TODO (enh): implement a recovery

        if do_recovery:
            result = self._recover_result(prelim_result=prelim_result)

        return result
    def copy(self):
        return copy.copy(self)

    @_Minimizer_register_check_support(True)
    def _minimize(self, loss, params, init):
        raise MinimizeNotImplementedError

    def __str__(self) -> str:
        string = f'<{self.name} strategy={self.strategy} tol={self.tol}>'
        return string

    def get_maxiter(self, n):
        maxiter = self.maxiter
        if callable(maxiter):
            maxiter = maxiter(n)
        elif maxiter == 'auto':
            maxiter = self._n_iter_per_param * n
        return maxiter

    def create_evaluator(self, loss: ZfitLoss, params: ztyping.ParametersType) -> LossEval:
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             maxiter=self.get_maxiter(len(params)),
                             minimizer=self)
        return evaluator

    def _update_tol_inplace(self, criterion_value, internal_tol):
        tol_factor = min([max([self.tol / criterion_value * 0.3, 1e-2]), 0.2])
        for tol in internal_tol:
            if tol in ('gtol', 'xtol'):
                internal_tol[tol] *= math.sqrt(tol_factor)
            else:
                internal_tol[tol] *= tol_factor

    def create_criterion(self, loss, params):
        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        return criterion

    # TODO: implement a recovery by using a "stateful" minimization
    def _recover_result(self, prelim_result):
        warnings.warn("recovering result, yet no special functionality implemented yet.")
        return prelim_result


class BaseStepMinimizer(BaseMinimizer):

    @minimize_supports()
    def _minimize(self, loss, params, init):
        n_old_vals = 10
        changes = collections.deque(np.ones(n_old_vals))
        last_val = -10
        niter = 0
        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        criterion_val = self.tol * 1000
        converged = False
        cur_val = 'invalid'
        prelim_result = None
        maxiter = self.get_maxiter(len(params))
        while True:
            cur_val = run(self._step(loss=loss, params=params, init=prelim_result))
            niter += 1


            changes.popleft()
            changes.append(abs(cur_val - last_val))
            sum_changes = np.sum(changes)
            maxiter_reached = niter > maxiter
            if sum_changes < self.tol and niter % 3 or maxiter_reached:  # test the last time surely
                xvalues = np.array(run(params))
                hesse = run(loss.hessian(params))
                inv_hesse = np.linalg.inv(hesse)
                status = 10
                criterion_val = criterion.last_value
                params = {p: val for p, val in zip(params, xvalues)}

                info = {'success': False, 'message': 'Unfinished, for criterion',
                        'n_eval': niter, 'inv_hesse': inv_hesse}
                prelim_result = FitResult(params=params, edm=criterion_val, fmin=cur_val, info=info, converged=False,
                                          status=status, valid=False, message='not yet converged', niter=niter,
                                          loss=loss, minimizer=self)
                converged = criterion.converged(prelim_result)
                if converged or maxiter_reached:
                    break

            last_val = cur_val

        # compose fit result
        message = "Maxiter reached" if maxiter_reached else ""

        success = criterion_val < self.tol
        status = 0 if success else 10
        info = {'success': success, 'message': message, 'n_eval': niter, 'inv_hesse': inv_hesse}

        params = {p: val for p, val in zip(params, xvalues)}
        valid = converged

        return FitResult(
            edm=criterion_val,
            message=message,
            niter=niter,
            valid=valid,
            params=params,
            criterion=criterion,
            fmin=cur_val,
            info=info,
            converged=success,
            status=status,
            loss=loss,
            minimizer=self.copy(),
        )

    def step(self, loss, params: ztyping.ParamsOrNameType = None, init: FitResult = None):
        """Perform a single step in the minimization (if implemented).

        Args:
            params:

        Returns:

        Raises:
            MinimizeStepNotImplementedError: if the `step` method is not implemented in the minimizer.
        """
        loss, params, init = self._check_convert_input(loss, params, init=init)

        return self._step(loss, params=params, init=init)

    def _step(self, loss, params, init):
        raise MinimizeStepNotImplementedError


class NOT_SUPPORTED:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Should never be instantated.")


def print_minimization_status(converged, criterion, evaluator, i, fmin,
                              internal_tol: Optional[Mapping[str, float]] = None):
    internal_tol = {} if internal_tol is None else internal_tol
    tols_str = ', '.join(f'{tol}={val:.3g}' for tol, val in internal_tol.items())
    print(f"{f'CONVERGED{os.linesep}' if converged else ''}"
          f"Finished iteration {i}, niter={evaluator.niter}, fmin={fmin:.7g},"
          f" {criterion.name}={criterion.last_value:.3g} {tols_str}")

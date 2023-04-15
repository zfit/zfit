#  Copyright (c) 2023 zfit
"""Definition of minimizers, wrappers etc."""
from __future__ import annotations

import collections
import copy
import functools
import inspect
import math
import os
import warnings
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager

import numpy as np
from ordered_set import OrderedSet

from .evaluation import LossEval
from .fitresult import FitResult
from .interface import ZfitMinimizer, ZfitResult
from .strategy import FailMinimizeNaN, PushbackStrategy, ZfitStrategy
from .termination import EDM, ConvergenceCriterion
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..core.parameter import assign_values, convert_to_parameters
from ..settings import run
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (
    InitNotImplemented,
    MaximumIterationReached,
    MinimizeNotImplemented,
    MinimizerSubclassingError,
    MinimizeStepNotImplemented,
    ParameterNotIndependentError,
)
from ..util.warnings import warn_changed_feature

DefaultStrategy = PushbackStrategy

status_messages = {"maxiter": "Maximum iteration reached."}


def minimize_supports(*, init: bool = False) -> Callable:
    """Decorator: Add (mandatory for some methods) on a method to control what it can handle.

    If any of the flags is set to False, it will check the arguments and, in case they match a flag
    (say if a *init* is passed while the *init* flag is set to ``False``), it will
    raise a corresponding exception (in this example a ``FromResultNotImplemented``) that will
    be caught by an outer function that knows how to handle things.

    Args:
        init: Specify whether the minimize method can handle a FitResult instead of a loss as a loss. There are
            three options:
            - False: This is the default and means that _no FitResult will ever come true_. The minimizer handles the
              initial parameter values himselves.
            - 'same': If 'same' is set, a ``FitResult`` will only come through if it was created with the *exact* same
              type as
        multiple_limits: If False, only simple limits are to be expected and no iteration is
            therefore required.
    """

    def wrapper(func):
        parameters = inspect.signature(func).parameters
        keys = list(parameters.keys())
        init_str = "init"
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
                    if init == "same":
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
    """Marks a method that the subclass either *has* to or *can't* use the ``@supports`` decorator.

    Args:
        has_support: If True, flags that it **requires** the ``@supports`` decorator. If False,
            flags that the ``@supports`` decorator is **not allowed**.
    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* ``support`` or (if False) has *no* ``support``.

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
        "tol": 1e-3,
        "verbosity": 0,
        "strategy": DefaultStrategy,
        "criterion": EDM,
        "maxiter": "auto",
    }

    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        criterion: ConvergenceCriterion | None = None,
        strategy: ZfitStrategy | None = None,
        minimizer_options: dict | None = None,
        maxiter: str | int | None = None,
        name: str | None = None,
    ) -> None:
        """Base Minimizer to minimize loss functions and return a result.

        This class acts as a base class to implement a minimizer. The method ``minimize`` has to be overridden.


        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            verbosity: |@doc:minimizer.verbosity| Verbosity of the minimizer. Has to be between 0 and 10.
              The verbosity has the meaning:

               - a value of 0 means quiet and no output
               - above 0 up to 5, information that is good to know but without
                 flooding the user, corresponding to a "INFO" level.
               - A value above 5 starts printing out considerably more and
                 is used more for debugging purposes.
               - Setting the verbosity to 10 will print out every
                 evaluation of the loss function and gradient.

               Some minimizers offer additional output which is also
               distributed as above but may duplicate certain printed values. |@docend:minimizer.verbosity|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            minimizer_options: Additional minimizer options
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__()
        self._n_iter_per_param = 3000

        self.tol = self._DEFAULTS["tol"] if tol is None else tol
        self.verbosity = self._DEFAULTS["verbosity"] if verbosity is None else verbosity
        self.minimizer_options = {} if minimizer_options is None else minimizer_options
        self.criterion = self._DEFAULTS["criterion"] if criterion is None else criterion

        if strategy is None:
            strategy = self._DEFAULTS["strategy"]
        try:
            do_error = not issubclass(strategy, ZfitStrategy)
        except TypeError:  # legacy
            warn_changed_feature(
                message="A strategy should now be a class, not an instance. The minimizer will"
                " at the beginning of the minimization create an instance that can be"
                " stateful during the minimization and will be stored in the FitResult.",
                identifier="strategies_in_minimizers.",
            )
            do_error = not isinstance(strategy, ZfitStrategy)
        if do_error:
            raise TypeError(f"strategy {strategy} is not a subclass of ZfitStrategy.")

        self._strategy = strategy
        self._state = None
        self.maxiter = self._DEFAULTS["maxiter"] if maxiter is None else maxiter
        self.name = repr(self.__class__)[:-2].split(".")[-1] if name is None else name

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if subclass has decorator if required
        cls._subclass_check_support(
            methods_to_check=_Minimizer_CHECK_HAS_SUPPORT,
            wrapper_not_overwritten=_Minimizer_register_check_support,
        )

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
                        raise MinimizerSubclassingError(
                            "Method {} has been wrapped with minimize_supports "
                            "but is not allowed to. Has to handle all "
                            "arguments.".format(method_name)
                        )
                elif has_support:
                    raise MinimizerSubclassingError(
                        f"Method {method_name} has been overwritten and *has to* be"
                        " wrapped by `@minimize_supports` decorator (don't forget"
                        " to call the decorator as it takes arguments)"
                    )
                elif not has_support:
                    continue  # no support, has not been wrapped with
            else:
                if not has_support:
                    continue  # not wrapped, no support, need no

            # if we reach this points, somethings was implemented wrongly
            raise MinimizerSubclassingError(
                f"Method {method_name} has not been correctly wrapped with "
                f"@minimize_supports "
            )

    def _check_convert_input(
        self, loss: ZfitLoss, params, init=None, floating=True
    ) -> tuple[ZfitLoss, Iterable[ZfitParameter], None | FitResult]:
        """Sanitize the input values and return all of them.

        Args:
            loss: If the loss is a callable, it will be converted to a SimpleLoss.
            params: If the parameters is an array, it will be converted to free parameters.
            init:
            floating:

        Returns:
            loss, params, init:
        """
        # TODO: cleanup logic of setting parameter values
        to_set_param_values = {}

        if isinstance(loss, ZfitResult):
            init = loss  # make the names correct
            loss = init.loss
            if params is None:
                params = list(init.params)
            elif not any(isinstance(p, ZfitParameter) for p in params):
                params_init = init.loss.get_params()
                to_set_param_values = {p: val for p, val in zip(params_init, params)}

        if isinstance(params, collections.abc.Mapping):
            if all(isinstance(p, ZfitParameter) for p in params):
                to_set_param_values = {
                    p: val for p, val in params.items() if val is not None
                }
                params = list(params.keys())
            elif all(isinstance(p, str) for p in params):
                params = convert_to_parameters(params, prefer_constant=False)
            else:
                raise ValueError(
                    "if `params` argument is a dict, it must either contain parameters or fields"
                    " such as value, name etc."
                )

        # convert the function to a SimpleLoss
        if not isinstance(loss, ZfitLoss):
            if not callable(loss):
                raise TypeError("Given Loss has to  be a ZfitLoss or a callable.")
            elif params is None:
                raise ValueError(
                    "If the loss is a callable, the params cannot be None."
                )

            from zfit.core.loss import SimpleLoss

            convert_to_parameters(params, prefer_constant=False)
            loss = SimpleLoss(func=loss, params=params)

        if isinstance(params, (tuple, list)) and not any(
            isinstance(p, ZfitParameter) for p in params
        ):
            loss_params = loss.get_params()
            if len(params) != len(loss_params):
                raise ValueError(
                    "params initial values have to have the same length as the free parameters of the loss:"
                    f" {len(params)} and {len(loss_params)} respectively."
                )
            to_set_param_values = {
                p: val for p, val in zip(loss_params, params) if val is not None
            }
            params = loss_params

        if params is None:
            params = loss.get_params(floating=floating)
        else:
            if to_set_param_values:
                try:
                    assign_values(
                        list(to_set_param_values), list(to_set_param_values.values())
                    )
                except ParameterNotIndependentError as error:
                    not_indep_and_set = {
                        p
                        for p, val in to_set_param_values.items()
                        if val is not None and not p.independent
                    }
                    raise ParameterNotIndependentError(
                        f"Cannot set parameter {not_indep_and_set} to a value as they"
                        f" are not independent. The following `param` argument was"
                        f" given: {params}."
                        f""
                        f"Original error"
                        f"--------------"
                        f"{error}"
                    ) from error
            else:
                params = convert_to_container(params, container=OrderedSet)

            # now extract all the independent parameters
            params = list(
                OrderedSet.union(*(p.get_params(floating=floating) for p in params))
            )

        # set the parameter values from the init
        if init is not None:
            # don't set the user set
            params_to_set = OrderedSet(params).intersection(
                OrderedSet(init.params)
            ) - OrderedSet(to_set_param_values)
            assign_values(params_to_set, init)
        if floating:
            params = self._filter_floating_params(params)
        if not params:
            raise RuntimeError(
                "No parameter for minimization given/found. Cannot minimize."
            )
        params = list(params)
        return loss, params, init

    @staticmethod
    def _filter_floating_params(params):
        non_floating = [param for param in params if not param.floating]
        if non_floating:  # legacy warning
            warnings.warn(
                f"CHANGED BEHAVIOR! Non-floating parameters {non_floating} will not be used in the "
                f"minimization."
            )
        return [param for param in params if param.floating]

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    def minimize(
        self,
        loss: ZfitLoss | Callable,
        params: ztyping.ParamsTypeOpt | None = None,
        init: ZfitResult | None = None,
    ) -> FitResult:
        """Fully minimize the `loss` with respect to `params`, optionally using information from `init`.

        The minimizer changes the parameter values in order to minimize the loss function until the convergence
        criterion value is less than the tolerance. This is a stateless function that can take a `FitResult` in order
        to initialize the minimization.

        Args:
            loss: Loss to be minimized until convergence is reached. Usually a :class:`ZfitLoss`.

            - If this is a simple callable that takes an array as argument and an attribute `errordef`. The attribute
              can be set to any arbitrary function like

              .. code-block::

                    def loss(x):
                        return - x ** 2

                    loss.errordef = 0.5  # as an example
                    minimizer.minimize(loss, [2, 5])

              If not TensorFlow is used inside the function, make sure to set `zfit.run.set_graph_mode(False)`
              and `zfit.run.set_autograd_mode(False)`.
            - A `FitResult` can be provided as the only argument to the method, in which case the loss as well as the
              parameters to be minimized are taken from it. This allows to easily chain minimization algorithms.

            params: The parameters with respect to which to
                minimize the `loss`. If `None`, the parameters will be taken from the `loss`.

                In order to fix the parameter values to a specific value (and thereby make them indepented
                of their current value), a dictionary mapping a parameter to a value can be given.

                If `loss` is a callable, `params` can also be (instead of `Parameters`):

                  - an array of initial values
                  - for more control, a ``dict`` with the keys:

                    - ``value`` (required): array-like initial values.
                    - ``name``: list of unique names of the parameters.
                    - ``lower``: array-like lower limits of the parameters,
                    - ``upper``: array-like upper limits of the parameters,
                    - ``step_size``: array-like initial step size of the parameters (approximately the expected
                      uncertainty)

                This will create internally a single parameter for each value that can be accessed in the `FitResult`
                via params. Repeated calls can therefore (in the current implement) cause a memory increase.
                The recommended way is to re-use parameters (just taken from the `FitResult` attribute `params`).
            init: A result of a previous minimization that provides auxiliary information such as the starting point for
                the parameters, the approximation of the covariance and more. Which information is used can depend on
                the specific minimizer implementation.

                In general, the assumption is that *the loss provided is similar enough* to the one provided in `init`.

                What is assumed to be close:

                - the parameters at the minimum of *loss* will be close to the parameter values at the minimum of
                  *init*.
                - Covariance matrix, or in general the shape, of *init* to the *loss* at its minimum.

                What is explicitly _not_ assumed to be the same:

                - absolute value of the loss function. If *init* has a function value at minimum x of fmin,
                  it is not assumed that `loss` will have the same/similar value at x.
                - parameters that are used in the minimization may differ in order or which are fixed.

        Returns:
            The fit result containing all information about the minimization.

        Examples:
            Using the ability to restart a minimization with a previous result allows to use a more global search
            algorithm with a high tolerance and an additional local minimization to polish the found minimum.

            .. code-block:: python

                result_approx = minimizer_global.minimize(loss, params)
                result = minimizer_local.minimize(result_approx)

            For a simple usage with a callable only, the parameters can be given as an array of initial values.

            .. code-block:: python

                def func(x):
                    return np.log(np.sum(x ** 2))

                func.errordef = 0.5
                params = [1.1, 3.5, 8.35]  # initial values
                result = minimizer.minimize(func, param)
        """

        loss, params, init = self._check_convert_input(
            loss=loss, params=params, init=init, floating=True
        )
        with self._make_stateful(loss=loss, params=params, init=init):
            return self._call_minimize(loss=loss, params=params, init=init)

    def _call_minimize(
        self,
        loss: ZfitLoss | Callable,
        params: ztyping.ParamsTypeOpt | None = None,
        init: ZfitResult | None = None,
    ) -> FitResult:
        do_recovery = False
        prelim_result = None

        try:
            result = self._minimize(loss=loss, params=params, init=init)
        except TypeError as error:
            if "got an unexpected keyword argument 'init'" in error.args[0]:
                warnings.warn(
                    "_minimize has to take an `init` argument. This will be mandatory in the future, please"
                    " change the signature accordingly.",
                    category=FutureWarning,
                    stacklevel=2,
                )
                result = self._call_minimize(loss=loss, params=params)
            else:
                raise
        except InitNotImplemented:
            assign_values(params=params, values=init)
            result = self._call_minimize(loss=loss, params=params)
        except (
            FailMinimizeNaN,
            RuntimeError,
        ):  # iminuit raises RuntimeError if user raises Error
            do_recovery = True
            strategy = self._state.get("strategy")
            if strategy is not None:
                prelim_result = strategy.fit_result
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

    @_Minimizer_register_check_support(True)
    def _minimize(
        self,
        loss: ZfitLoss | Callable,
        params: ztyping.ParamsTypeOpt | None = None,
        init: ZfitResult | None = None,
    ) -> FitResult:
        raise MinimizeNotImplemented

    @property
    def _is_stateful(self):
        return self._state is not None

    @contextmanager
    def _make_stateful(
        self,
        loss: ZfitLoss | Callable,
        params: ztyping.ParamsTypeOpt | None = None,
        init: ZfitResult | None = None,
    ) -> None:
        """Remember the loss, param and init that is currently used inside the minimization.

        Args:
            loss: Loss to be minimized. Can be a simple callable that takes an array as
            params: The parameters with respect to which to
                minimize the `loss`. If `None`, the parameters will be taken from the `loss`.
            init: A result of a previous minimization that provides auxiliary information such as the starting point for
                the parameters
        """
        state = {"loss": loss, "params": params, "init": init}
        self._state = state
        yield
        self._state = None

    def copy(self):
        return copy.copy(self)

    def __str__(self) -> str:
        return f"<{type(self).__name__} {self.name} tol={self.tol}>"

    def get_maxiter(self, n=None):
        if n is None:
            if self._is_stateful:
                n = len(self._state["params"])
            else:
                raise ValueError("n cannot be None if not called within minimize")
        maxiter = self.maxiter
        if callable(maxiter):
            maxiter = maxiter(n)
        elif maxiter == "auto":
            maxiter = self._n_iter_per_param * n
        return maxiter

    def create_evaluator(
        self,
        loss: ZfitLoss | None = None,
        params: ztyping.ParametersType | None = None,
        numpy_converter: Callable | None = None,
        strategy: ZfitStrategy | None = None,
    ) -> LossEval:
        """Make a loss evaluator using the strategy and more from the minimizer.

        Convenience factory for the loss evaluator.
        This wraps the loss to return a numpy array, to catch NaNs, stop on maxiter and evaluate the gradient
        and hessian without the need to specify the order every time.

        Args:
            loss: Loss to be wrapped. Can be None if called inside `_minimize`
            params: Parameters that will be associated with the loss in this order. Can be None if called within
                `_minimize`.
            strategy: Instance of a Strategy that will be used during the evaluation.

        Returns:
            LossEval: The evaluator that wraps the Loss ant Strategy with the current parameters.
        """
        if loss is None:
            if self._is_stateful:
                loss = self._state["loss"]
            else:
                raise ValueError("loss cannot be None if not called within minimize")

        if params is None:
            if self._is_stateful:
                params = self._state["params"]
            else:
                raise ValueError("params cannot be None if not called within minimize")

        if numpy_converter is None:
            numpy_converter = False
        if strategy is None:
            try:
                strategy = self._strategy()
                if not isinstance(strategy, ZfitStrategy):
                    raise TypeError
            except TypeError:  # cannot be called -> is not a class LEGACY
                strategy = self._strategy

        if self._is_stateful:
            self._state["strategy"] = strategy
        evaluator = LossEval(
            loss=loss,
            params=params,
            strategy=strategy,
            do_print=self.verbosity > 9,
            maxiter=self.get_maxiter(len(params)),
            numpy_converter=numpy_converter,
        )
        if self._is_stateful:
            self._state["evaluator"] = evaluator
        return evaluator

    def _update_tol_inplace(self, criterion_value, internal_tol):
        tol_factor = min(
            math.sqrt(min([max([self.tol / criterion_value * 0.3, 1e-4]), 0.04])), 0.21
        )
        for tol in internal_tol:
            if tol in ("gtol", "xtol"):
                internal_tol[tol] *= math.sqrt(tol_factor)
            else:
                internal_tol[tol] *= tol_factor

    def create_criterion(
        self,
        loss: ZfitLoss | None = None,
        params: ztyping.ParametersType | None = None,
    ) -> ConvergenceCriterion:
        """Create a criterion instance for the given loss and parameters.

        Args:
            loss: Loss that is used for the criterion. Can be None if called inside `_minimize`
            params: Parameters that will be associated with the loss in this order. Can be None if called within
                `_minimize`.

        Returns:
            ConvergenceCriterion to check if the function converged.
        """
        if loss is None:
            if self._is_stateful:
                loss = self._state["loss"]
            else:
                raise ValueError("loss cannot be None if not called within minimize")

        if params is None:
            if self._is_stateful:
                params = self._state["params"]
            else:
                raise ValueError("params cannot be None if not called within minimize")

        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        if self._is_stateful:
            self._state["criterion"] = criterion
        return criterion

    # TODO: implement a recovery by using a "stateful" minimization
    def _recover_result(self, prelim_result):
        warnings.warn(
            "recovering result, yet no special functionality implemented yet."
        )
        return prelim_result


BaseMinimizerV1 = BaseMinimizer


class BaseStepMinimizer(BaseMinimizer):
    """Step minimizer that uses the `_step` method to advance a single step and check if the criterion is reached.py.

    In order to subclass this correctly, override `_step`.
    """

    @minimize_supports()
    def _minimize(self, loss, params, init):
        if init:
            assign_values(params=params, values=init)
        n_old_vals = 5
        changes = collections.deque(np.ones(n_old_vals))
        last_val = -10
        niter = 0
        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        prelim_result = None
        maxiter = self.get_maxiter(len(params))
        criterion_val = None
        while True:
            cur_val = run(self._step(loss=loss, params=params, init=prelim_result))
            niter += 1

            changes.popleft()
            changes.append(abs(cur_val - last_val))
            sum_changes = np.sum(changes)
            maxiter_reached = niter > maxiter
            if (
                sum_changes < self.tol and niter % 3
            ) or maxiter_reached:  # test the last time surely
                xvalues = np.array(run(params))
                hesse = run(loss.hessian(params))
                inv_hesse = np.linalg.inv(hesse)
                status = 10
                params_result = {p: val for p, val in zip(params, xvalues)}

                message = "Unfinished, for criterion"
                info = {
                    "success": False,
                    "message": message,
                    "n_eval": niter,
                    "inv_hesse": inv_hesse,
                }
                prelim_result = FitResult(
                    params=params_result,
                    edm=criterion_val,
                    fmin=cur_val,
                    info=info,
                    converged=False,
                    status=status,
                    valid=False,
                    message=message,
                    niter=niter,
                    criterion=criterion,
                    loss=loss,
                    minimizer=self,
                )
                converged = criterion.converged(prelim_result)
                criterion_val = criterion.last_value

                if converged or maxiter_reached:
                    break

            last_val = cur_val

        # compose fit result
        message = "Maxiter reached" if maxiter_reached else ""

        success = converged
        status = 0 if success else 10
        info = {
            "success": success,
            "message": message,
            "n_eval": niter,
            "inv_hesse": inv_hesse,
        }

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

    def step(
        self, loss, params: ztyping.ParamsOrNameType = None, init: FitResult = None
    ):
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
        raise MinimizeStepNotImplemented


class NOT_SUPPORTED:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Should never be instantated.")


def print_minimization_status(
    converged,
    criterion,
    evaluator,
    i,
    fmin,
    internal_tol: Mapping[str, float] | None = None,
):
    internal_tol = {} if internal_tol is None else internal_tol
    tols_str = ", ".join(f"{tol}={val:.3g}" for tol, val in internal_tol.items())
    print(
        f"{f'CONVERGED{os.linesep}' if converged else ''}"
        f"Finished iteration {i}, niter={evaluator.niter}, fmin={fmin:.7g},"
        f" {criterion.name}={criterion.last_value:.3g} {tols_str}"
    )

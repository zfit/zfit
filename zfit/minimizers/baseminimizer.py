"""
Definition of minimizers, wrappers etc.

"""

#  Copyright (c) 2021 zfit
import abc
import collections
import copy
import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import List, Union, Iterable, Mapping

import numpy as np
import texttable as tt

from .fitresult import FitResult
from .interface import ZfitMinimizer, ZfitResult
from .termination import EDM
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..core.parameter import set_values
from ..settings import run
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import MinimizeNotImplementedError, MinimizeStepNotImplementedError


class FailMinimizeNaN(Exception):
    pass


class ZfitStrategy(abc.ABC):
    @abstractmethod
    def minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                     values: Mapping = None) -> float:
        raise NotImplementedError


class BaseStrategy(ZfitStrategy):

    def __init__(self) -> None:
        self.fit_result = None
        self.error = None
        super().__init__()

    def minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                     values: Mapping = None) -> float:
        return self._minimize_nan(loss=loss, params=params, minimizer=minimizer, values=values)

    def _minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                      values: Mapping = None) -> (float, np.array):
        print("The minimization failed due to too many NaNs being produced in the loss."
              "This is most probably caused by negative"
              " values returned from the PDF. Changing the initial values/stepsize of the parameters can solve this"
              " problem. Also check your model (if custom) for problems. For more information,"
              " visit https://github.com/zfit/zfit/wiki/FAQ#fitting-and-minimization")
        raise FailMinimizeNaN()

    def __str__(self) -> str:
        return repr(self.__class__)[:-2].split(".")[-1]


class ToyStrategyFail(BaseStrategy):

    def __init__(self) -> None:
        super().__init__()
        self.fit_result = FitResult(params={}, edm=-999, fmin=-999, status=-999, converged=False, info={},
                                    loss=None, minimizer=None)

    def _minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                      values: Mapping = None) -> float:
        param_vals = run(params)
        param_vals = OrderedDict((param, value) for param, value in zip(params, param_vals))
        self.fit_result = FitResult(params=param_vals, edm=-999, fmin=-999, status=9, converged=False, info={},
                                    loss=loss,
                                    minimizer=minimizer)
        raise FailMinimizeNaN()


class PushbackStrategy(BaseStrategy):

    def __init__(self, nan_penalty: Union[float, int] = 100, nan_tolerance: int = 30, **kwargs):
        """Pushback by adding `nan_penalty * counter` to the loss if NaNs are encountered.

        The counter indicates how many NaNs occurred in a row. The `nan_tolerance` is the upper limit, if this is
        exceeded, the fallback will be used and an error is raised.

        Args:
            nan_penalty: Value to add to the previous loss in order to penalize the step taken.
            nan_tolerance: If the number of NaNs encountered in a row exceeds this number, the fallback is used.
        """
        super().__init__(**kwargs)
        self.nan_penalty = nan_penalty
        self.nan_tolerance = nan_tolerance

    def _minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                      values: Mapping = None) -> float:
        assert 'nan_counter' in values, "'nan_counter' not in values, minimizer not correctly implemented"
        nan_counter = values['nan_counter']
        if nan_counter < self.nan_tolerance:
            last_loss = values.get('old_loss')
            last_grad = values.get('old_grad')
            if last_grad is not None:
                last_grad = -last_grad
            if last_loss is not None:

                loss_evaluated = last_loss + self.nan_penalty * nan_counter
            else:
                loss_evaluated = values.get('loss')
            if isinstance(loss_evaluated, str):
                raise RuntimeError("Loss starts already with NaN, cannot minimize.")
            return loss_evaluated, last_grad
        else:
            super()._minimize_nan(loss=loss, params=params, minimizer=minimizer, values=values)


DefaultStrategy = PushbackStrategy


class DefaultToyStrategy(DefaultStrategy, ToyStrategyFail):
    """Same as :py:class:`DefaultStrategy`, but does not raise an error on full failure, instead return an invalid
    FitResult.

    This can be useful for toy studies, where multiple fits are done and a failure should simply be counted as a
    failure instead of rising an error.
    """


class BaseMinimizer(ZfitMinimizer):
    """Minimizer for loss functions.

    Additional `minimizer_options` (given as **kwargs) can be accessed and changed via the
    attribute (dict) `minimizer.minimizer_options`.
    """
    _DEFAULT_TOLERANCE = 1e-3

    def __init__(self, name, tolerance, verbosity, minimizer_options, strategy=None, maxiter=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = repr(self.__class__)[:-2].split(".")[-1]
        if strategy is None:
            strategy = DefaultStrategy()
        if not isinstance(strategy, ZfitStrategy):
            raise TypeError(f"strategy {strategy} is not an instance of ZfitStrategy.")
        self.strategy = strategy
        self.name = name
        if tolerance is None:
            tolerance = self._DEFAULT_TOLERANCE
        self.tolerance = tolerance
        self.verbosity = 5 if verbosity is None else verbosity
        if minimizer_options is None:
            minimizer_options = {}
        self.minimizer_options = minimizer_options
        self.maxiter = 5000 if maxiter is None else maxiter
        self._max_steps = 5000
        self._convergence_criterion_cls = EDM

    def _check_input_params(self, loss: ZfitLoss, params, only_floating=True):

        params = convert_to_container(params)
        if params is None:
            params = loss.get_params(only_floating=only_floating)
            params = list(params)
        else:
            params_indep = []
            for param in params:
                if param.independent:
                    params_indep.append(param)
                else:
                    params_indep.extend(param.get_params(only_floating=only_floating))
            params = params_indep

        if only_floating:
            params = self._filter_floating_params(params)
        if not params:
            raise RuntimeError("No parameter for minimization given/found. Cannot minimize.")
        return params

    @staticmethod
    def _filter_floating_params(params):
        non_floating = [param for param in params if not param.floating]
        if non_floating:  # legacy warning
            warnings.warn(f"CHANGED BEHAVIOR! Non-floating parameters {non_floating} will not be used in the "
                          f"minimization.")
        return [param for param in params if param.floating]

    @staticmethod
    def _extract_load_method(params):
        return [param.load for param in params]

    @staticmethod
    def _extract_param_names(params):
        return [param.name for param in params]

    def _check_gradients(self, params, gradients):
        non_dependents = [param for param, grad in zip(params, gradients) if grad is None]
        if non_dependents:
            raise ValueError("Invalid gradients for the following parameters: {}"
                             "The function does not depend on them. Probably a Tensor"
                             "instead of a `CompositeParameter` was created implicitly.".format(non_dependents))

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance

    @staticmethod
    def _extract_start_values(params):
        """Extract the current value if defined, otherwise random.

        Arguments:
            params:

        Return:
            list(const): the current value of parameters
        """
        values = [p for p in params]
        return values

    @staticmethod
    def _update_params(params: Union[Iterable[ZfitParameter]], values: Union[Iterable[float], np.ndarray]) -> List[
        ZfitParameter]:
        """Update `params` with `values`. Returns the assign op (if `use_op`, otherwise use a session to load the value.

        Args:
            params: The parameters to be updated
            values: New values for the parameters.

        Returns:
            List of assign operations if `use_op`, otherwise empty. The output
                can therefore be directly used as argument to :py:func:`~tf.control_dependencies`.
        """
        if len(params) == 1 and len(values) > 1:
            values = (values,)  # iteration will be correctly
        for param, value in zip(params, values):
            param.set_value(value)

        return params


    def minimize(self, loss: ZfitLoss, params: ztyping.ParamsTypeOpt = None) -> FitResult:
        """Fully minimize the `loss` with respect to `params`.

        Args:
            loss: Loss to be minimized.
            params: The parameters with respect to which to
                minimize the `loss`. If `None`, the parameters will be taken from the `loss`.

        Returns:
            The fit result.
        """

        result = None
        if isinstance(loss, ZfitResult):
            result = loss  # make the names correct
            loss = result.loss
        params = self._check_input_params(loss=loss, params=params, only_floating=True)
        if result is not None:
            set_values(params, result)

        try:
            return self._hook_minimize(loss=loss, params=params)
        except (FailMinimizeNaN, RuntimeError) as error:  # iminuit raises RuntimeError if user raises Error
            fail_result = self.strategy.fit_result
            if fail_result is not None:
                return fail_result
            else:
                raise

    def _hook_minimize(self, loss, params):
        return self._call_minimize(loss=loss, params=params)

    def _call_minimize(self, loss, params):
        try:
            return self._minimize(loss=loss, params=params)
        except MinimizeNotImplementedError as error:
            try:
                return self._minimize_with_step(loss=loss, params=params)
            except MinimizeStepNotImplementedError:
                raise error

    def copy(self):
        return copy.copy(self)

    def _minimize(self, loss, params):
        raise MinimizeNotImplementedError



    def __str__(self) -> str:
        string = f'<{self.name} strategy={self.strategy} tolerance={self.tolerance}>'
        return string


class BaseStepMinimizer(BaseMinimizer):

    def _minimize(self, loss, params):
        n_old_vals = 10
        changes = collections.deque(np.ones(n_old_vals))
        last_val = -10
        n_steps = 0
        criterion = self._convergence_criterion_cls(tolerance=self.tolerance, loss=loss, params=params)
        edm = self.tolerance * 1000
        sum_changes = np.sum(changes)
        inv_hesse = None
        while (sum_changes > self.tolerance or edm > self.tolerance) and n_steps < self._max_steps:
            cur_val = run(self.step(loss=loss, params=params))
            if sum_changes < self.tolerance and n_steps % 5:
                xvalues = np.array(run(params))
                hesse = run(loss.hessian(params))
                inv_hesse = np.linalg.inv(hesse)
                edm = criterion.calculateV1(value=cur_val, xvalues=xvalues, grad=run(loss.gradients(params)),
                                            inv_hesse=inv_hesse)
            changes.popleft()
            changes.append(abs(cur_val - last_val))
            sum_changes = np.sum(changes)
            last_val = cur_val
            n_steps += 1
        fmin = cur_val

        # compose fit result
        message = "successful finished"
        are_unique = len(set([run(change) for change in changes])) > 1  # values didn't change...
        if not are_unique:
            message = "Loss unchanged for last {} steps".format(n_old_vals)

        success = are_unique
        status = 0 if success else 10
        xvalues = np.array(run(params))
        info = {'success': success, 'message': message, 'n_eval': n_steps, 'inv_hesse': inv_hesse}

        params = OrderedDict((p, val) for p, val in zip(params, xvalues))

        return FitResult(params=params, edm=edm, fmin=fmin, info=info,
                         converged=success, status=status,
                         loss=loss, minimizer=self.copy())


    def step(self, loss, params: ztyping.ParamsOrNameType = None):
        """Perform a single step in the minimization (if implemented).

        Args:
            params:

        Returns:

        Raises:
            MinimizeStepNotImplementedError: if the `step` method is not implemented in the minimizer.
        """
        params = self._check_input_params(loss, params)

        return self._step(loss, params=params)


    def _step(self, loss, params):
        raise MinimizeStepNotImplementedError


def print_params(params, values, loss=None):
    table = tt.Texttable()
    table.header(['Parameter', 'Value'])

    for param, value in zip(params, values):
        table.add_row([param.name, value])
    if loss is not None:
        table.add_row(["Loss value:", loss])
    print(table.draw())


def print_gradients(params, values, gradients, loss=None):
    table = tt.Texttable()
    table.header(['Parameter', 'Value', 'Gradient'])
    for param, value, grad in zip(params, values, gradients):
        table.add_row([param.name, value, grad])
    if loss is not None:
        table.add_row(["Loss value:", loss, "|"])
    print(table.draw())

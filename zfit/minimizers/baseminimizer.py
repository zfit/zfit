"""
Definition of minimizers, wrappers etc.

"""

#  Copyright (c) 2020 zfit
import abc
import collections
import copy
from abc import abstractmethod
from collections import OrderedDict
from typing import List, Union, Iterable, Mapping

import numpy as np
import texttable as tt

from .fitresult import FitResult
from .interface import ZfitMinimizer
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..settings import run
from ..util import ztyping




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
                      values: Mapping = None) -> float:
        print("The minimization failed due to NaNs being produced in the loss. This is most probably caused by negative"
              " values returned from the PDF. Changing the initial values/stepsize of the parameters can solve this"
              " problem. Also check your model (if custom) for problems. For more information,"
              " visit https://github.com/zfit/zfit/wiki/FAQ#fitting-and-minimization")
        raise FailMinimizeNaN()


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


class DefaultStrategy(BaseStrategy):

    def __init__(self) -> None:
        super().__init__()
        self._nan_penalty = 10000

    def _minimize_nan(self, loss: ZfitLoss, params: ztyping.ParamTypeInput, minimizer: ZfitMinimizer,
                     values: Mapping = None) -> float:
        import zfit
        if zfit.experimental_loss_penalty_nan:
            last_loss = values.get('old_loss')
            loss_evaluated = last_loss + self._nan_penalty if last_loss is not None else values.get('loss')
            return loss_evaluated
        else:
            super()._minimize_nan(loss=loss, params=params, minimizer=minimizer, values=values)


class BaseMinimizer(ZfitMinimizer):
    """Minimizer for loss functions.

    Additional `minimizer_options` (given as **kwargs) can be accessed and changed via the
    attribute (dict) `minimizer.minimizer_options`

    """
    _DEFAULT_name = "BaseMinimizer"
    _DEFAULT_TOLERANCE = 1e-3

    def __init__(self, name, tolerance, verbosity, minimizer_options, strategy=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = self._DEFAULT_name
        if strategy is None:
            strategy = DefaultStrategy()
        if not isinstance(strategy, ZfitStrategy):
            raise TypeError(f"strategy {strategy} is not an instance of ZfitStrategy.")
        self.strategy = strategy
        self.name = name
        if tolerance is None:
            tolerance = self._DEFAULT_TOLERANCE
        self.tolerance = tolerance
        self.verbosity = verbosity
        if minimizer_options is None:
            minimizer_options = {}
        self.minimizer_options = minimizer_options

    def _check_input_params(self, loss: ZfitLoss, params, only_floating=True):
        if isinstance(params, (str, ZfitParameter)) or (not hasattr(params, "__len__") and params is not None):
            params = [params, ]
            params = self._filter_floating_params(params)
        if params is None or isinstance(params[0], str):
            params = loss.get_dependents(only_floating=only_floating)
            params = list(params)
        if not params:
            raise RuntimeError("No parameter for minimization given/found. Cannot minimize.")
        return params

    @staticmethod
    def _filter_floating_params(params):
        params = [param for param in params if param.floating]
        return params

    @staticmethod
    def _extract_load_method(params):
        params_load = [param.load for param in params]
        return params_load

    @staticmethod
    def _extract_param_names(params):
        names = [param.name for param in params]
        return names

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
            params (Parameter):

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
            params (list(`ZfitParameter`)): The parameters to be updated
            values (list(float, `np.ndarray`)): New values for the parameters.

        Returns:
            list(empty, :py:class:`~tf.Operation`): List of assign operations if `use_op`, otherwise empty. The output
                can therefore be directly used as argument to :py:func:`~tf.control_dependencies`.
        """
        if len(params) == 1 and len(values) > 1:
            values = (values,)  # iteration will be correctly
        for param, value in zip(params, values):
            param.set_value(value)

        return params

    def step(self, loss, params: ztyping.ParamsOrNameType = None):
        """Perform a single step in the minimization (if implemented).

        Args:
            params ():

        Returns:

        Raises:
            NotImplementedError: if the `step` method is not implemented in the minimizer.
        """
        params = self._check_input_params(loss, params)

        return self._step(loss, params=params)

    def minimize(self, loss: ZfitLoss, params: ztyping.ParamsTypeOpt = None) -> FitResult:
        """Fully minimize the `loss` with respect to `params`.

        Args:
            loss (ZfitLoss): Loss to be minimized.
            params (list(`zfit.Parameter`): The parameters with respect to which to
                minimize the `loss`. If `None`, the parameters will be taken from the `loss`.

        Returns:
            `FitResult`: The fit result.
        """
        params = self._check_input_params(loss=loss, params=params, only_floating=True)
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
        except NotImplementedError as error:
            try:
                return self._minimize_with_step(loss=loss, params=params)
            except NotImplementedError:
                raise error

    def _minimize_with_step(self, loss, params):  # TODO improve
        n_old_vals = 10
        changes = collections.deque(np.ones(n_old_vals))
        last_val = -10

        def step_fn(loss, params):
            try:
                self._step_tf(loss=loss.value, params=params)
            except NotImplementedError:
                self.step(loss, params)
            return loss.value()

        while sum(sorted(changes)[-3:]) > self.tolerance:  # TODO: improve condition
            cur_val = step_fn(loss=loss, params=params)
            changes.popleft()
            changes.append(abs(cur_val - last_val))
            last_val = cur_val
        fmin = cur_val
        edm = -999  # TODO: get edm

        # compose fit result
        message = "successful finished"
        are_unique = len(set([float(change.numpy()) for change in changes])) > 1  # values didn't change...
        if not are_unique:
            message = "Loss unchanged for last {} steps".format(n_old_vals)

        success = are_unique
        status = 0 if success else 10

        info = {'success': success, 'message': message}  # TODO: create status
        param_values = [float(p.numpy()) for p in params]
        params = OrderedDict((p, val) for p, val in zip(params, param_values))

        return FitResult(params=params, edm=edm, fmin=fmin, info=info,
                         converged=success, status=status,
                         loss=loss, minimizer=self.copy())

    def copy(self):
        return copy.copy(self)


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
        table.add_row(["Loss value:", loss])
    print(table.draw())

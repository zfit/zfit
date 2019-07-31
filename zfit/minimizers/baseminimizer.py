"""
Definition of minimizers, wrappers etc.

"""

#  Copyright (c) 2019 zfit

import collections
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
import copy
from typing import List, Union

import numpy as np
import tensorflow as tf

from ..settings import run
from .interface import ZfitMinimizer
from ..util.execution import SessionHolderMixin
from .fitresult import FitResult
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..util import ztyping


class FailMinimizeNaN(Exception):
    pass


class ZfitStrategy(metaclass=ABCMeta):

    @abstractmethod
    def minimize_nan(self, loss, loss_value, params):
        raise NotImplementedError


class BaseStrategy(ZfitStrategy):

    def __init__(self) -> None:
        self.fit_result = None
        self.error = None
        super().__init__()

    def minimize_nan(self, loss, params, minimizer, loss_value=None, gradient_values=None):
        return self._minimize_nan(loss=loss, params=params, minimizer=minimizer, loss_value=loss_value,
                                  gradient_values=gradient_values)

    def _minimize_nan(self, loss, loss_value, params, minimizer, gradient_values):
        raise NotImplementedError


class ToyStrategyFail(BaseStrategy):

    def _minimize_nan(self, loss, params, minimizer, loss_value, gradient_values):
        values = run(params)
        params = OrderedDict((param, value) for param, value in zip(params, values))
        self.fit_result = FitResult(params=params, edm=-999, fmin=-999, status=9, converged=False, info={}, loss=loss,
                                    minimizer=minimizer)
        raise FailMinimizeNaN()


class DefaultStrategy(BaseStrategy):

    def _minimize_nan(self, loss, params, minimizer, loss_value, gradient_values):
        pass  # nothing to do here


class BaseMinimizer(SessionHolderMixin, ZfitMinimizer):
    """Minimizer for loss functions.

    Additional `minimizer_options` (given as **kwargs) can be accessed and changed via the
    attribute (dict) `minimizer.minimizer_options`

    """
    _DEFAULT_name = "BaseMinimizer"

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
            tolerance = 1e-5
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
    def _update_params(params: Union[List[ZfitParameter]], values: Union[List[float], np.ndarray],
                       use_op: bool = False) -> List[tf.Operation]:
        """Update `params` with `values`. Returns the assign op (if `use_op`, otherwise use a session to load the value.

        Args:
            params (list(`ZfitParameter`)): The parameters to be updated
            values (list(float, `np.ndarray`)): New values for the parameters.
            use_op (bool): Use the :py:meth:`~tf.Variable.assign` operation and return a list of them. If
                False, use  :py:meth:`~tf.Variable.load` and a session to load the values

        Returns:
            list(empty, :py:class:`~tf.Operation`): List of assign operations if `use_op`, otherwise empty. The output
                can therefore be directly used as argument to :py:func:`~tf.control_dependencies`.
        """
        new_params_op = []
        if len(params) == 1 and len(values) > 1:
            values = (values,)  # iteration will be correctly
        for param, value in zip(params, values):
            if use_op:
                new_params_op.append(param.assign(value))
            else:
                param.set_value(value)

        return new_params_op

    def step(self, loss, params: ztyping.ParamsOrNameType = None):
        """Perform a single step in the minimization (if implemented).

        Args:
            params ():

        Returns:

        Raises:
            NotImplementedError: if the `step` method is not implemented in the minimizer.
        """
        params = self._check_input_params(params)
        with ExitStack() as stack:
            tuple(stack.enter_context(param.set_sess(self.sess)) for param in params)

            return self._step(params=params)

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
        with ExitStack() as stack:
            tuple(stack.enter_context(param.set_sess(self.sess)) for param in params)
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
        try:
            step = self._step_tf(loss=loss, params=params)
        except NotImplementedError:
            step_fn = self.step
        else:
            def step_fn(loss, params):
                return self.sess.run([step, loss.value()])

        while sum(sorted(changes)[-3:]) > self.tolerance:  # TODO: improve condition
            _, cur_val = step_fn(loss=loss, params=params)
            changes.popleft()
            changes.append(abs(cur_val - last_val))
            last_val = cur_val
        fmin = cur_val
        edm = -999  # TODO: get edm

        # compose fit result
        message = "successful finished"
        are_unique = len(set(changes)) > 1  # values didn't change...
        if not are_unique:
            message = "Loss unchanged for last {} steps".format(n_old_vals)

        success = are_unique
        status = 0 if success else 10

        info = {'success': success, 'message': message}  # TODO: create status
        param_values = self.sess.run(params)
        params = OrderedDict((p, val) for p, val in zip(params, param_values))

        return FitResult(params=params, edm=edm, fmin=fmin, info=info,
                         converged=success, status=status,
                         loss=loss, minimizer=self.copy())

    def copy(self):
        return copy.copy(self)

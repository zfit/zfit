#  Copyright (c) 2021 zfit
import contextlib
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import texttable as tt

from ..core.interfaces import ZfitLoss
from ..core.parameter import assign_values, set_values
from ..settings import run
from ..util import ztyping
from ..util.exception import MaximumIterationReached
from .strategy import ZfitStrategy


class LossEval:

    def __init__(self,
                 loss: ZfitLoss,
                 params: ztyping.ParamTypeInput,
                 strategy: ZfitStrategy,
                 do_print: bool,
                 maxiter: int,
                 grad_fn: Optional[Callable] = None,
                 hesse_fn: Optional[Callable] = None):
        r"""Convenience wrapper for the evaluation of a loss with given parameters and strategy.

        The methods `value`, `gradient` etc will raise a `MaximumIterationReached` error in case the maximum iterations
        is reached.

        Args:
            loss: Loss that will be used to be evaluated.
            params: Parameters that the gradient and hessian will be derived to.
            strategy: A strategy to deal with NaNs and more.
            do_print: If the values should be printed nicely
            maxiter: Maximum number of evaluations of the `value`, 'gradient` or `hessian`.
            grad_fn: Function that returns the gradient.
            hesse_fn: Function that returns the hessian matrix.
        """
        super().__init__()
        self.maxiter = maxiter
        self._ignoring_maxiter = False
        with self.ignore_maxiter():  # otherwise when trying to set, it gets all of them, which fails as not there
            self.nfunc_eval = 0
            self.ngrad_eval = 0
            self.nhess_eval = 0

        self.loss = loss
        if hesse_fn is None:
            hesse_fn = loss.hessian

        self.hesse_fn = hesse_fn
        if grad_fn is not None:
            def value_gradients_fn(params):
                return loss.value(), grad_fn(params)
        else:
            value_gradients_fn = self.loss.value_gradient
            grad_fn = self.loss.gradient
        self.gradients_fn = grad_fn
        self.value_gradients_fn = value_gradients_fn
        self.loss = loss
        self.last_value = None
        self.last_gradient = None
        self.last_hessian = None
        self.nan_counter = 0
        self.params = params
        self.strategy = strategy
        self.do_print = do_print

    @property
    def niter(self):
        return max([self.nfunc_eval, self.ngrad_eval, self.nhess_eval])

    @property
    def maxiter_reached(self):
        return not self.ignoring_maxiter and self.niter > self.maxiter

    def _check_maxiter_reached(self):
        if self.maxiter_reached:
            raise MaximumIterationReached

    @contextlib.contextmanager
    def ignore_maxiter(self):
        """Return temporary the maximum number of iterations and won't raise an error."""
        old = self._ignoring_maxiter
        self._ignoring_maxiter = True
        yield
        self._ignoring_maxiter = old

    @property
    def ignoring_maxiter(self):
        return self._ignoring_maxiter or self.maxiter is None

    @property
    def nfunc_eval(self):
        return self._nfunc_eval

    @nfunc_eval.setter
    def nfunc_eval(self, value):
        self._nfunc_eval = value
        self._check_maxiter_reached()

    @property
    def ngrad_eval(self):
        return self._ngrad_eval

    @ngrad_eval.setter
    def ngrad_eval(self, value):
        self._ngrad_eval = value
        self._check_maxiter_reached()

    @property
    def nhess_eval(self):
        return self._nhess_eval

    @nhess_eval.setter
    def nhess_eval(self, value):
        self._nhess_eval = value
        self._check_maxiter_reached()

    def value_gradient(self, values: np.ndarray) -> Tuple[np.float64, np.ndarray]:
        """Calculate the value and gradient like :py:meth:`~ZfitLoss.value_gradients`.

        Args:
            values: parameter values to calculate the value and gradient at.

        Returns:
            Tuple[numpy.float64, numpy.ndarray]:

        Raises:
            MaximumIterationReached: If the maximum number of iterations (including tolerance) was reached.
        """
        if not self._ignoring_maxiter:
            self.nfunc_eval += 1
            self.ngrad_eval += 1

        assign_values(self.params, values=values)
        is_nan = False

        try:
            loss_value, gradient = self.value_gradients_fn(params=self.params)
            loss_value = run(loss_value)
            gradient_values = np.array(run(gradient))
            loss_value, gradient_values, _ = self.strategy.callback(value=loss_value,
                                                                    gradient=gradient_values,
                                                                    hessian=None,
                                                                    params=self.params,
                                                                    loss=self.loss)
        except Exception as error:
            loss_value = "invalid, error occured"
            gradient_values = ["invalid"] * len(self.params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:

            if self.do_print:
                try:
                    print_gradient(self.params, values, gradient=gradient_values, loss=loss_value)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or any(np.isnan(gradient_values)) or np.isnan(loss_value)
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': loss_value,
                'grad': gradient_values,
                'old_loss': self.last_value,
                'old_grad': self.last_gradient,
                'nan_counter': self.nan_counter,
            }

            loss_value, gradient_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                                     values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
            self.last_gradient = gradient_values
        return loss_value, gradient_values

    def value(self, values: np.ndarray) -> np.float64:
        """Calculate the value like :py:meth:`~ZfitLoss.value`.

        Args:
            values: parameter values to calculate the value at.

        Returns:
            Calculated loss value

        Raises:
            MaximumIterationReached: If the maximum number of iterations (including tolerance) was reached.
        """
        if not self._ignoring_maxiter:
            self.nfunc_eval += 1
        assign_values(self.params, values=values)
        is_nan = False

        try:
            loss_value = self.loss.value()
            loss_value = run(loss_value)
            loss_value, _, _ = self.strategy.callback(value=loss_value,
                                                      gradient=None,
                                                      hessian=None,
                                                      params=self.params,
                                                      loss=self.loss)
        except Exception as error:
            loss_value = "invalid, error occured"
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                try:
                    print_params(self.params, values, loss=loss_value)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or np.isnan(loss_value).any()
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': loss_value,
                'old_loss': self.last_value,
                'nan_counter': self.nan_counter,
            }

            loss_value, _ = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                       values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
        return loss_value

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """Calculate the gradient like :py:meth:`~ZfitLoss.gradient`.

        Args:
            values: parameter values to calculate the value and gradient at.

        Returns:
            Tuple[numpy.float64, numpy.ndarray]:

        Raises:
            MaximumIterationReached: If the maximum number of iterations (including tolerance) was reached.
        """
        if not self._ignoring_maxiter:
            self.ngrad_eval += 1
        assign_values(self.params, values=values)
        is_nan = False

        try:
            gradient = self.gradients_fn(params=self.params)
            gradient_values = np.array(run(gradient))
            _, gradient_values, _ = self.strategy.callback(value=None,
                                                           gradient=gradient_values,
                                                           hessian=None,
                                                           params=self.params,
                                                           loss=self.loss)
        except Exception as error:
            gradient_values = ["invalid"] * len(self.params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                try:
                    print_gradient(self.params, values, gradient=gradient_values, loss=-999)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or any(np.isnan(gradient_values))
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': -999,
                'grad': gradient_values,
                'old_loss': self.last_value,
                'old_grad': self.last_gradient,
                'nan_counter': self.nan_counter,
            }

            _, gradient_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                            values=info_values)
        else:
            self.nan_counter = 0
            self.last_gradient = gradient_values
        return gradient_values

    def hessian(self, values) -> np.ndarray:
        """Calculate the hessian like :py:meth:`~ZfitLoss.hessian`.

        Args:
            values: parameter values to calculate the hessian at.

        Returns:
            Hessian matrix

        Raises:
            MaximumIterationReached: If the maximum number of iterations (including tolerance) was reached.
        """
        if not self._ignoring_maxiter:
            self.nhess_eval += 1
        assign_values(self.params, values=values)
        is_nan = False

        try:
            hessian = self.hesse_fn(params=self.params)
            hessian_values = np.array(run(hessian))
            _, _, hessian = self.strategy.callback(value=None,
                                                   gradient=None,
                                                   hessian=hessian,
                                                   params=self.params,
                                                   loss=self.loss)
        except Exception as error:
            hessian_values = ["invalid"] * len(self.params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                try:
                    print_params(self.params, values, loss=-999)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or np.any(np.isnan(hessian_values))
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': -999,
                'old_loss': self.last_value,
                'old_grad': self.last_gradient,
                'nan_counter': self.nan_counter,
            }

            _, _ = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                              values=info_values)
        else:
            self.nan_counter = 0
            self.last_hessian = hessian_values
        return hessian_values


def print_params(params, values, loss=None):
    table = tt.Texttable()
    table.header(['Parameter', 'Value'])

    for param, value in zip(params, values):
        table.add_row([param.name, value])
    if loss is not None:
        table.add_row(["Loss value:", loss])
    print(table.draw())


def print_gradient(params, values, gradient, loss=None):
    table = tt.Texttable()
    table.header(['Parameter', 'Value', 'Gradient'])
    for param, value, grad in zip(params, values, gradient):
        table.add_row([param.name, value, grad])
    if loss is not None:
        table.add_row(["Loss value:", loss, "|"])
    print(table.draw())

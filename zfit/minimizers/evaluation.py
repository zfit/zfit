#  Copyright (c) 2021 zfit
import contextlib
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
import texttable as tt

from .strategy import ZfitStrategy
from .interface import ZfitMinimizer
from ..core.interfaces import ZfitLoss
from ..core.parameter import set_values
from ..settings import run
from ..util import ztyping
from ..util.exception import MaximumIterationReached


class LossEval:

    def __init__(self,
                 loss: ZfitLoss,
                 params: ztyping.ParamTypeInput,
                 strategy: ZfitStrategy,
                 do_print: bool,
                 minimizer: ZfitMinimizer,
                 maxiter: int,
                 grad_fn: Optional[Callable] = None,
                 hesse_fn: Optional[Callable] = None,
                 niter_tol: Optional[float] = None):
        super().__init__()
        niter_tol = 0.1 if niter_tol is None else niter_tol
        self.niter_tol = niter_tol
        self.maxiter = maxiter
        self._ignoring_maxiter = False
        with self.ignore_maxiter():  # otherwise when trying to set, it gets all of them, which fails as not there
            self.nfunc_eval = 0
            self.ngrad_eval = 0
            self.nhess_eval = 0

        self.loss = loss
        self.minimizer = minimizer
        if hesse_fn is None:
            hesse_fn = loss.hessian

        self.hesse_fn = hesse_fn
        if grad_fn is not None:
            def value_gradients_fn(params):
                return loss.value(), grad_fn(params)
        else:
            value_gradients_fn = self.loss.value_gradients
            grad_fn = self.loss.gradients
        self.gradients_fn = grad_fn
        self.value_gradients_fn = value_gradients_fn
        self.maxiter_reached = False
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

    def _check_maxiter(self):
        if not self.ignoring_maxiter and self.niter * (1 + self.niter_tol) > self.maxiter:
            raise MaximumIterationReached

    @contextlib.contextmanager
    def ignore_maxiter(self):
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
        self._check_maxiter()

    @property
    def ngrad_eval(self):
        return self._ngrad_eval

    @ngrad_eval.setter
    def ngrad_eval(self, value):
        self._ngrad_eval = value
        self._check_maxiter()

    @property
    def nhess_eval(self):
        return self._nhess_eval

    @nhess_eval.setter
    def nhess_eval(self, value):
        self._nhess_eval = value
        self._check_maxiter()

    def value_gradients(self, values):
        self.nfunc_eval += 1
        self.ngrad_eval += 1

        set_values(self.params, values=values)
        is_nan = False

        try:
            loss_value, gradients = self.value_gradients_fn(params=self.params)
            loss_value = run(loss_value)
            gradients_values = np.array(run(gradients))
        except Exception as error:
            loss_value = "invalid, error occured"
            gradients_values = ["invalid"] * len(self.params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                try:
                    print_gradients(self.params, values, gradients=gradients_values, loss=loss_value)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or any(np.isnan(gradients_values)) or np.isnan(loss_value)
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': loss_value,
                'grad': gradients_values,
                'old_loss': self.last_value,
                'old_grad': self.last_gradient,
                'nan_counter': self.nan_counter,
            }

            loss_value, gradients_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                                      minimizer=self.minimizer,
                                                                      values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
            self.last_gradient = gradients_values
        return loss_value, gradients_values

    def value(self, values):
        self.nfunc_eval += 1
        set_values(self.params, values=values)
        is_nan = False

        try:
            loss_value = self.loss.value()
            loss_value = run(loss_value)
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

        is_nan = is_nan or np.isnan(loss_value)
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': loss_value,
                'old_loss': self.last_value,
                'nan_counter': self.nan_counter,
            }

            loss_value, _ = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                       minimizer=self.minimizer,
                                                       values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
        return loss_value

    def gradient(self, values):
        self.ngrad_eval += 1
        set_values(self.params, values=values)
        is_nan = False

        try:
            gradients = self.gradients_fn(params=self.params)
            gradients_values = np.array(run(gradients))
        except Exception as error:
            gradients_values = ["invalid"] * len(self.params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                try:
                    print_gradients(self.params, values, gradients=gradients_values, loss=-999)
                except:
                    print("Cannot print loss value or gradient values.")

        is_nan = is_nan or any(np.isnan(gradients_values))
        if is_nan:
            self.nan_counter += 1
            info_values = {
                'loss': -999,
                'grad': gradients_values,
                'old_loss': self.last_value,
                'old_grad': self.last_gradient,
                'nan_counter': self.nan_counter,
            }

            _, gradients_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                             minimizer=self.minimizer,
                                                             values=info_values)
        else:
            self.nan_counter = 0
            self.last_gradient = gradients_values
        return gradients_values

    def hessian(self, values):
        self.nhess_eval += 1
        set_values(self.params, values=values)
        is_nan = False

        try:
            hessian = self.hesse_fn(params=self.params)
            hessian_values = np.array(run(hessian))
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
                                              minimizer=self.minimizer,
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


def print_gradients(params, values, gradients, loss=None):
    table = tt.Texttable()
    table.header(['Parameter', 'Value', 'Gradient'])
    for param, value, grad in zip(params, values, gradients):
        table.add_row([param.name, value, grad])
    if loss is not None:
        table.add_row(["Loss value:", loss, "|"])
    print(table.draw())

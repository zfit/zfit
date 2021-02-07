#  Copyright (c) 2021 zfit
from typing import Callable

import numpy as np
import tensorflow as tf

from .baseminimizer import print_gradients, ZfitStrategy, print_params
from .interface import ZfitMinimizer
from ..core.interfaces import ZfitLoss
from ..core.parameter import set_values
from ..settings import run
from ..util import ztyping


class LossEval:

    def __init__(self,
                 loss: ZfitLoss,
                 params: ztyping.ParamTypeInput,
                 strategy: ZfitStrategy,
                 do_print: bool,
                 minimizer: ZfitMinimizer,
                 grad_fn: Callable = None,
                 hesse_fn: Callable = None):
        super().__init__()
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
        self.loss = loss
        self.current_grad_value = None
        self.current_loss_value = None
        self.nan_counter = 0
        self.params = params
        self.strategy = strategy
        self.do_print = do_print

    def value_gradients(self, values):

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
                'old_loss': self.current_loss_value,
                'old_grad': self.current_grad_value,
                'nan_counter': self.nan_counter,
            }

            loss_value, gradients_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                                      minimizer=self.minimizer,
                                                                      values=info_values)
        else:
            self.nan_counter = 0
            self.current_loss_value = loss_value
            self.current_grad_value = gradients_values
        return loss_value, gradients_values

    def value(self, values):

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
                'old_loss': self.current_loss_value,
                'nan_counter': self.nan_counter,
            }

            loss_value, _ = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                       minimizer=self.minimizer,
                                                       values=info_values)
        else:
            self.nan_counter = 0
            self.current_loss_value = loss_value
        return loss_value

    def gradient(self, values):

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
                'old_loss': self.current_loss_value,
                'old_grad': self.current_grad_value,
                'nan_counter': self.nan_counter,
            }

            _, gradients_values = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                                             minimizer=self.minimizer,
                                                             values=info_values)
        else:
            self.nan_counter = 0
            self.current_grad_value = gradients_values
        return gradients_values

    def hessian(self, values):

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
                'old_loss': self.current_loss_value,
                'old_grad': self.current_grad_value,
                'nan_counter': self.nan_counter,
            }

            _, _ = self.strategy.minimize_nan(loss=self.loss, params=self.params,
                                              minimizer=self.minimizer,
                                              values=info_values)
        else:
            self.nan_counter = 0
        return hessian_values

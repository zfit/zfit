#  Copyright (c) 2024 zfit

from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import partial

import numpy as np
import tensorflow as tf
import texttable as tt

from ..core.interfaces import ZfitLoss
from ..core.parameter import assign_values_jit
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import DerivativeCalculationError, MaximumIterationReached
from ..z import numpy as znp
from .strategy import ZfitStrategy


def assign_values_func(params, values):
    # TODO(WrappedVariable): this is needed if we want to use wrapped Variables
    # params = z.math._extract_tfparams(params)
    return assign_values_jit(params, znp.asarray(values))


def check_derivative_none_raise(values, params) -> None:
    """Check if values contains ``None`` and raise a ``DerivativeCalculationError`` if so.

    Args:
        values: Values to check. Can be gradient or hessian
        params: Parameter that correspond to the values.
    """
    if None in values:
        none_params = [p for p, grad in zip(params, values) if grad is None]
        msg = (
            f"The derivative of the following parameters is None: {none_params}."
            f" This is usually caused by either the function not depending on the"
            f" parameter (or not in a differentiable way) or by using pure Python"
            f" code instead of TensorFlow."
        )
        raise DerivativeCalculationError(msg)


class LossEval:
    def __init__(
        self,
        loss: ZfitLoss,
        params: ztyping.ParamTypeInput,
        strategy: ZfitStrategy,
        do_print: bool,
        maxiter: int,
        grad_fn: Callable | None = None,
        hesse_fn: Callable | None = None,
        numpy_converter: Callable | None = None,
        full: bool | None = None,
    ):
        r"""Convenience wrapper for the evaluation of a loss with given parameters and strategy.

        The methods ``value``, ``gradient`` etc will raise a ``MaximumIterationReached`` error in case the maximum iterations
        is reached.

        Args:
            loss: Loss that will be used to be evaluated.
            params: Parameters that the gradient and hessian will be derived to.
            strategy: A strategy to deal with NaNs and more.
            do_print: If the values should be printed nicely
            maxiter: Maximum number of evaluations of the ``value``, 'gradient`` or ``hessian``.
            grad_fn: Function that returns the gradient.
            hesse_fn: Function that returns the hessian matrix.
            numpy_converter: Converter to a numpy-like format. For example, ``np.array`` will create
                a *writable* numpy array, whereas ``np.asarray`` will create a *read-only* numpy array.
                Useful if the return values should be numpy arrays (and not "numpy-like" objects). This is needed
                if the function expects something *writeable* like a numpy array as other (JAX, TensorFlow) arrays
                are not writeable. If None, no conversion is done and a "nmupy-like" object is returned.
            full: If True, the full loss will be calculated. Default is False. For the minimization, the full loss is usually not needed.
        """
        super().__init__()
        full = False if full is None else full
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
                return loss.value(full=full), grad_fn(params)

        else:
            value_gradients_fn = partial(self.loss.value_gradient, full=full)
            grad_fn = self.loss.gradient
        self.gradients_fn = grad_fn
        self.value_gradients_fn = value_gradients_fn
        self.loss = loss
        self.last_value = None
        self.last_gradient = None
        self.last_hessian = None
        self.nan_counter = 0
        self.params = convert_to_container(params)
        self.strategy = strategy
        self.do_print = do_print
        self.full = full
        self.numpy_converter = False if numpy_converter is None else numpy_converter

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

    def value_gradient(self, values: np.ndarray) -> tuple[np.float64, np.ndarray]:
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

        params = self.params
        assign_values_func(params, values=values)
        is_nan = False

        try:
            loss_value, gradient = self.value_gradients_fn(params=params)
            loss_value, gradient, _ = self.strategy.callback(
                value=loss_value,
                gradient=gradient,
                hessian=None,
                params=params,
                loss=self.loss,
            )
        except Exception as error:
            loss_value = "invalid, error occured"
            gradient = ["invalid"] * len(params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                print_gradient(params, values, gradient=gradient, loss=loss_value)

        check_derivative_none_raise(gradient, params)
        is_nan = is_nan or any(znp.isnan(gradient)) or znp.isnan(loss_value)
        if is_nan:
            self.nan_counter += 1
            info_values = {
                "loss": loss_value,
                "grad": gradient,
                "old_loss": self.last_value,
                "old_grad": self.last_gradient,
                "nan_counter": self.nan_counter,
            }

            loss_value, gradient = self.strategy.minimize_nan(loss=self.loss, params=params, values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
            self.last_gradient = gradient
        if self.numpy_converter:
            loss_value = float(loss_value)
            gradient = self.numpy_converter(gradient)
        return loss_value, gradient

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
        params = self.params
        assign_values_func(params, values=values)
        is_nan = False
        loss_value = None
        try:
            loss_value = self.loss.value(full=self.full)
            loss_value, _, _ = self.strategy.callback(
                value=loss_value,
                gradient=None,
                hessian=None,
                params=params,
                loss=self.loss,
            )
        except Exception as error:
            loss_value = "invalid, error occured"
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                print_params(params, values, loss=loss_value)

        is_nan = is_nan or np.isnan(loss_value).any()
        if is_nan:
            self.nan_counter += 1
            info_values = {
                "loss": loss_value,
                "old_loss": self.last_value,
                "nan_counter": self.nan_counter,
            }

            loss_value, _ = self.strategy.minimize_nan(loss=self.loss, params=params, values=info_values)
        else:
            self.nan_counter = 0
            self.last_value = loss_value
        if self.numpy_converter:
            loss_value = float(loss_value)
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
        params = self.params
        assign_values_func(params, values=values)
        is_nan = False
        gradient = None
        try:
            gradient = self.gradients_fn(params=params)
            _, gradient, _ = self.strategy.callback(
                value=None,
                gradient=gradient,
                hessian=None,
                params=params,
                loss=self.loss,
            )
        except Exception as error:
            gradient = ["invalid"] * len(params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                print_gradient(params, values, gradient=gradient, loss=None)

        check_derivative_none_raise(gradient, params)

        is_nan = is_nan or znp.any(znp.isnan(gradient))
        if is_nan:
            self.nan_counter += 1
            info_values = {
                "loss": None,
                "grad": gradient,
                "old_loss": self.last_value,
                "old_grad": self.last_gradient,
                "nan_counter": self.nan_counter,
            }

            _, gradient = self.strategy.minimize_nan(loss=self.loss, params=params, values=info_values)
        else:
            self.nan_counter = 0
            self.last_gradient = gradient
        if self.numpy_converter:
            gradient = self.numpy_converter(gradient)
        return gradient

    def hessian(self, values) -> znp.array:
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
        params = self.params
        assign_values_func(params, values=values)
        is_nan = False
        hessian = None
        try:
            hessian = self.hesse_fn(params=params)
            _, _, hessian = self.strategy.callback(
                value=None,
                gradient=None,
                hessian=hessian,
                params=params,
                loss=self.loss,
            )
        except Exception as error:
            hessian = ["invalid"] * len(params)
            if isinstance(error, tf.errors.InvalidArgumentError):
                is_nan = True
            else:
                raise

        finally:
            if self.do_print:
                print_params(params, values, loss=None)

        check_derivative_none_raise(hessian, params)

        is_nan = is_nan or znp.any(znp.isnan(hessian))
        if is_nan:
            self.nan_counter += 1
            info_values = {
                "loss": None,
                "old_loss": self.last_value,
                "old_grad": self.last_gradient,
                "nan_counter": self.nan_counter,
            }

            _, _ = self.strategy.minimize_nan(loss=self.loss, params=params, values=info_values)
        else:
            self.nan_counter = 0
            self.last_hessian = hessian
        if self.numpy_converter:
            hessian = self.numpy_converter(hessian)
        return hessian


def print_params(params, values, loss=None):
    table = tt.Texttable(max_width=0)

    row1 = []
    row2 = []
    if loss is not None:
        loss = float(loss)
        row1.append("Loss")
        row2.append(loss)

    # for param, value in zip(params, values):
    table.header(row1 + ["Parameter: "] + [param.label for param in params])
    table.add_row([*row2, "value: ", *list(values)])


def print_gradient(params, values, gradient, loss=None):
    table = tt.Texttable(max_width=0)

    header = []
    valrow = []
    gradrow = []
    if loss is not None:
        loss = float(loss)
        header.append("Loss")
        valrow.append(loss)
        gradrow.append("")

    table.header(header + ["Parameter"] + [param.name for param in params])
    table.add_row([*valrow, "Value:", *list(values)])
    table.add_row([*gradrow, "Gradient:", *list(gradient)])

#  Copyright (c) 2020 zfit

from typing import List, Optional

import iminuit
import numpy as np
import tensorflow as tf

from .baseminimizer import BaseMinimizer, ZfitStrategy, print_params, print_gradients
from .fitresult import FitResult
from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter
from ..util.cache import GraphCachable


class Minuit(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "Minuit"

    def __init__(self, strategy: ZfitStrategy = None, minimize_strategy: int = 1, tolerance: float = None,
                 verbosity: int = 5, name: str = None,
                 ncall: Optional[int] = None, use_minuit_grad: bool = None, **minimizer_options):
        """

        Args:
            strategy: A :py:class:`~zfit.minimizer.baseminimizer.ZfitStrategy` object that defines the behavior of
            the minimizer in certain situations.
            minimize_strategy: A number used by minuit to define the strategy, either 0, 1 or 2.
            tolerance: Stopping criteria: the Estimated Distance to Minimum (EDM) has to be lower then `tolerance`
            verbosity: Regulates how much will be printed during minimization. Values between 0 and 10 are valid.
            name: Name of the minimizer
            ncall: Maximum number of minimization steps.
            use_minuit_grad: If True, iminuit uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
        """
        minimizer_options['ncall'] = 0 if ncall is None else ncall
        if minimize_strategy not in range(3):
            raise ValueError(f"minimize_strategy has to be 0, 1 or 2, not {minimize_strategy}.")
        minimizer_options['strategy'] = minimize_strategy

        super().__init__(name=name, strategy=strategy, tolerance=tolerance, verbosity=verbosity,
                         minimizer_options=minimizer_options)
        use_minuit_grad = True if use_minuit_grad is None else use_minuit_grad
        self._minuit_minimizer = None
        self._use_tfgrad = not use_minuit_grad

    def _minimize(self, loss: ZfitLoss, params: List[Parameter]):

        # create options
        minimizer_options = self.minimizer_options.copy()
        minimize_options = {}
        minimize_options['precision'] = minimizer_options.pop('precision', None)
        minimize_options['ncall'] = minimizer_options.pop('ncall')

        minimizer_init = {}
        if 'errordef' in minimizer_options:
            raise ValueError("errordef cannot be specified for Minuit as this is already defined in the Loss.")
        loss_errordef = loss.errordef
        if not isinstance(loss_errordef, (float, int)):
            loss_errordef = 1.0  # default of minuit
        minimizer_init['errordef'] = loss_errordef
        minimizer_init['pedantic'] = minimizer_options.pop('pedantic', False)

        minimizer_setter = {}
        minimizer_setter['strategy'] = minimizer_options.pop('strategy')
        if self.verbosity > 6:
            minuit_verbosity = 3
        elif self.verbosity > 2:
            minuit_verbosity = 1
        else:
            minuit_verbosity = 0
        if minimizer_options:
            raise ValueError("The following options are not (yet) supported: {}".format(minimizer_options))

        # create Minuit compatible names
        limits = tuple(tuple((param.lower, param.upper)) for param in params)
        errors = tuple(param.step_size for param in params)
        start_values = [p.numpy() for p in params]
        limits = [(low.numpy(), up.numpy()) for low, up in limits]
        errors = [err.numpy() for err in errors]

        multiparam = isinstance(start_values[0], np.ndarray) and len(start_values[0]) > 1 and len(params) == 1
        if multiparam:
            # TODO(Mayou36): multiparameter
            params_name = None  # autogenerate for the moment
            start_values = start_values[0]
            errors = errors[0]
            limits = limits[0]
            gradients = gradients[0]
        else:
            params_name = [param.name for param in params]

        current_loss = None
        nan_counter = 0

        def func(values):
            nonlocal current_loss, nan_counter
            self._update_params(params=params, values=values)
            do_print = self.verbosity > 8

            is_nan = False

            try:
                loss_evaluated = loss.value().numpy()
            except tf.errors.InvalidArgumentError:
                is_nan = True
                loss_evaluated = "invalid, error occured"
            except:
                loss_evaluated = "invalid, error occured"
                raise
            finally:
                if do_print:
                    print_params(params, values, loss_evaluated)
            is_nan = is_nan or np.isnan(loss_evaluated)
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values['loss'] = loss_evaluated
                info_values['old_loss'] = current_loss
                info_values['nan_counter'] = nan_counter
                loss_evaluated = self.strategy.minimize_nan(loss=loss, params=params, minimizer=minimizer,
                                                            values=info_values)
            else:
                nan_counter = 0
                current_loss = loss_evaluated
            return loss_evaluated

        def grad_func(values):
            nonlocal current_loss, nan_counter
            self._update_params(params=params, values=values)
            do_print = self.verbosity > 8
            is_nan = False

            try:
                loss_value, gradients = loss.value_gradients(params=params)
                loss_value = loss_value.numpy()
                gradients_values = [float(g.numpy()) for g in gradients]
            except tf.errors.InvalidArgumentError:
                is_nan = True
                loss_value = "invalid, error occured"
                gradients_values = ["invalid"] * len(params)
            except:
                gradients_values = ["invalid"] * len(params)
                raise
            finally:
                if do_print:
                    print_gradients(params, values, gradients_values, loss=loss_value)

            is_nan = is_nan or any(np.isnan(gradients_values))
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values['loss'] = loss_value
                info_values['old_loss'] = current_loss
                info_values['nan_counter'] = nan_counter
                # but loss value not needed here
                _ = self.strategy.minimize_nan(loss=loss, params=params, minimizer=minimizer,
                                               values=info_values)
            else:
                nan_counter = 0
                current_loss = loss_value
            return gradients_values

        grad_func = grad_func if self._use_tfgrad else None

        minimizer = iminuit.Minuit.from_array_func(fcn=func, start=start_values,
                                                   error=errors, limit=limits, name=params_name,
                                                   grad=grad_func,
                                                   # use_array_call=True,
                                                   print_level=minuit_verbosity,
                                                   # forced_parameters=[f"param_{i}" for i in range(len(start_values))],
                                                   **minimizer_init)

        strategy = minimizer_setter.pop('strategy')
        minimizer.set_strategy(strategy)
        minimizer.tol = self.tolerance / 1e-3  # iminuit 1e-3 and tolerance 0.1
        assert not minimizer_setter, "minimizer_setter is not empty, bug. Please report. minimizer_setter: {}".format(
            minimizer_setter)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(**minimize_options)
        fitresult = FitResult.from_minuit(loss=loss, params=params, result=result, minimizer=self.copy())
        return fitresult

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer

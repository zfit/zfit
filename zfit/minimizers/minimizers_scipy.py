#  Copyright (c) 2021 zfit
from typing import Optional, Dict

import numpy as np
import tensorflow as tf

from .baseminimizer import BaseMinimizer, print_gradients, ZfitStrategy, print_params
from ..core.parameter import set_values
from ..settings import run


class Scipy(BaseMinimizer):

    def __init__(self, minimizer: str = 'L-BFGS-B', tolerance: Optional[float] = None,
                 strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5,
                 name: Optional[str] = None, num_grad: Optional[bool] = None,
                 minimizer_options: Optional[Dict[str, object]] = None):
        """SciPy optimizer algorithms.

        This is a wrapper for all the SciPy optimizers. More information can be found in their docs.

        Args:
            minimizer: Name of the minimization algorithm to use.
            tolerance: Stopping criterion of the algorithm to determine when the minimum
                has been found. The default is 1e-4, which is *different* from others.
            verbosity:             name: Name of the minimizer
            num_grad: If True, SciPy uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
            name: Name of the minimizer
            minimizer_options:
        """
        num_grad = False if num_grad is None else num_grad
        minimizer_options = {} if minimizer_options is None else minimizer_options
        if tolerance is None:
            tolerance = 1e-4
        if name is None:
            name = minimizer
        self.num_grad = num_grad
        minimizer_options = minimizer_options.copy()
        minimizer_options.update(method=minimizer)
        super().__init__(tolerance=tolerance, name=name, verbosity=verbosity,
                         strategy=strategy,
                         minimizer_options=minimizer_options)

    def _minimize(self, loss, params):

        current_loss = None
        nan_counter = None

        minimizer_options = self.minimizer_options.copy()

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

        def grad_value_func(values):
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
                    try:
                        print_gradients(params, values, gradients_values, loss=loss_value)
                    except:
                        print("Cannot print loss value or gradient values.")

            is_nan = is_nan or any(np.isnan(gradients_values)) or np.isnan(loss_value)
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values['loss'] = loss_value
                info_values['old_loss'] = current_loss
                info_values['nan_counter'] = nan_counter
                # but loss value not needed here
                _ = self.strategy.minimize_nan(loss=loss, params=params, minimizer=self,
                                               values=info_values)
            else:
                nan_counter = 0
                current_loss = loss_value
            return loss_value, gradients_values

        initial_values = np.array(run(params))
        limits = [(run(p.lower), run(p.upper)) for p in params]
        minimize_kwargs = {
            'jac': not self.num_grad,
            # 'callback': step_callback,
            'method': minimizer_options.pop('method'),
            # 'constraints': constraints,
            'bounds': limits,
            'tol': self.tolerance,
            'options': minimizer_options.pop('options', None),
        }
        minimize_kwargs.update(self.minimizer_options)
        import scipy.optimize  # pylint: disable=g-import-not-at-top
        result = scipy.optimize.minimize(fun=func if self.num_grad else grad_value_func,
                                         x0=initial_values, **minimize_kwargs)

        set_values(params, result['x'])
        from .fitresult import FitResult
        fitresult = FitResult.from_scipy(loss=loss, params=params, result=result, minimizer=self)

        return fitresult

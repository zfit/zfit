#  Copyright (c) 2022 zfit

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from .baseminimizer import BaseMinimizer, minimize_supports
from .evaluation import print_gradient
from .fitresult import FitResult
from .strategy import ZfitStrategy
from ..core.parameter import assign_values


class BFGS(BaseMinimizer):
    def __init__(
        self,
        strategy: ZfitStrategy = None,
        tol: float = 1e-5,
        verbosity: int = 5,
        max_calls: int = 3000,
        name: str = "BFGS_TFP",
        options: Mapping = None,
    ) -> None:
        """# Todo write description for api.

        Args:
            strategy: Strategy that handles NaN and more (to come, experimental)
            tol: Difference between the function value that suffices to stop minimization
            verbosity: The higher, the more is printed. Between 1 and 10 typically
            max_calls: Maximum number of calls, approximate
            name: Name of the Minimizer
            options: A `dict` containing the options given to the minimization function, overriding the default
        """
        self.options = {} if options is None else options
        self.max_calls = max_calls
        super().__init__(
            strategy=strategy,
            tol=tol,
            verbosity=verbosity,
            name=name,
            minimizer_options={},
            criterion=None,
            maxiter=None,
        )

    @minimize_supports()
    def _minimize(self, loss, params):
        from .. import run

        minimizer_fn = tfp.optimizer.bfgs_minimize
        params = tuple(params)
        do_print = self.verbosity > 8

        current_loss = None
        nan_counter = 0

        # @z.function
        def update_params_value_grad(loss, params, values):
            for param, value in zip(params, tf.unstack(values, axis=0)):
                param.set_value(value)
            value, gradients = loss.value_gradient(params=params)
            return gradients, value

        def to_minimize_func(values):
            nonlocal current_loss, nan_counter
            do_print = self.verbosity > 8

            is_nan = False
            gradient = None
            value = None
            try:
                gradient, value = update_params_value_grad(loss, params, values)

            except tf.errors.InvalidArgumentError:
                err = "NaNs"
                is_nan = True
            except:
                err = "unknonw error"
                raise
            finally:
                if value is None:
                    value = f"invalid, {err}"
                if gradient is None:
                    gradient = [f"invalid, {err}"] * len(params)
                if do_print:
                    print_gradient(
                        params,
                        run(values),
                        [float(run(g)) for g in gradient],
                        loss=run(value),
                    )
            loss_evaluated = run(value)
            is_nan = is_nan or np.isnan(loss_evaluated)
            if is_nan:
                nan_counter += 1
                info_values = {}
                info_values["loss"] = run(value)
                info_values["old_loss"] = current_loss
                info_values["nan_counter"] = nan_counter
                value = self.strategy.minimize_nan(
                    loss=loss, params=params, minimizer=self, values=info_values
                )
            else:
                nan_counter = 0
                current_loss = value

            gradient = znp.stack(gradient)
            return value, gradient

        initial_inv_hessian_est = tf.linalg.tensor_diag([p.step_size for p in params])

        minimizer_kwargs = dict(
            initial_position=znp.stack(params),
            x_tol=self.tol,
            # f_relative_tolerance=self.tolerance * 1e-5,  # TODO: use edm for stopping criteria
            initial_inverse_hessian_estimate=initial_inv_hessian_est,
            parallel_iterations=1,
            max_iterations=self.max_calls,
        )
        minimizer_kwargs.update(self.options)
        result = minimizer_fn(to_minimize_func, **minimizer_kwargs)

        # save result
        params_result = run(result.position)
        assign_values(params, values=params_result)

        info = {
            "n_eval": run(result.num_objective_evaluations),
            "n_iter": run(result.num_iterations),
            "grad": run(result.objective_gradient),
            "original": result,
        }
        edm = None
        fmin = run(result.objective_value)
        status = None
        converged = run(result.converged)
        params = OrderedDict((p, val) for p, val in zip(params, params_result))
        result = FitResult(
            params=params,
            edm=edm,
            fmin=fmin,
            info=info,
            loss=loss,
            status=status,
            converged=converged,
            minimizer=self.copy(),
        )
        return result

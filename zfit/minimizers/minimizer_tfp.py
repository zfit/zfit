#  Copyright (c) 2020 zfit
from collections import OrderedDict
from typing import Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .baseminimizer import BaseMinimizer, print_gradients, ZfitStrategy
from .fitresult import FitResult


class BFGS(BaseMinimizer):

    def __init__(self, strategy: ZfitStrategy = None, tolerance: float = 1e-5, verbosity: int = 5,
                 name: str = "BFGS_TFP", options: Mapping = None) -> None:
        """

        Args:
            strategy (ZfitStrategy): Strategy that handles NaN and more (to come, experimental)
            tolerance (float): Difference between the function value that suffices to stop minimization
            verbosity: The higher, the more is printed. Between 1 and 10 typically
            name: Name of the Minimizer
            options: A `dict` containing the options given to the minimization function, overriding the default
        """
        self.options = {} if options is None else options
        super().__init__(strategy=strategy, tolerance=tolerance, verbosity=verbosity, name=name,
                         minimizer_options={})

    def _minimize(self, loss, params):
        minimizer_fn = tfp.optimizer.bfgs_minimize
        params = tuple(params)
        do_print = self.verbosity > 5

        @tf.function(autograph=False, experimental_relax_shapes=True)
        def update_params_value_grad(loss, params, values):
            for param, value in zip(params, tf.unstack(values, axis=0)):
                param.set_value(value)
            value, gradients = loss.value_gradients(params=params)
            return gradients, value

        def to_minimize_func(values):
            gradients, value = update_params_value_grad(loss, params, values)
            if do_print:
                print_gradients(params, values.numpy(), [float(g.numpy()) for g in gradients])

            loss_evaluated = value.numpy()
            if np.isnan(loss_evaluated):
                self.strategy.minimize_nan(loss=loss, params=params, minimizer=None, values=loss_evaluated)
            gradients = tf.stack(gradients)
            return value, gradients

        initial_inv_hessian_est = tf.linalg.tensor_diag([p.step_size for p in params])

        minimizer_kwargs = dict(
            initial_position=tf.stack(params),
            # tolerance=1e-4,
            f_relative_tolerance=self.tolerance * 1e-3,  # TODO: use edm for stopping criteria
            initial_inverse_hessian_estimate=initial_inv_hessian_est,
            parallel_iterations=1,
            max_iterations=300
        )
        minimizer_kwargs.update(self.options)

        result = minimizer_fn(to_minimize_func,
                              **minimizer_kwargs)

        # save result
        params_result = result.position.numpy()
        self._update_params(params, values=params_result)

        info = {'n_eval': result.num_objective_evaluations.numpy(),
                'n_iter': result.num_iterations.numpy(),
                'grad': result.objective_gradient.numpy(),
                'original': result}
        edm = -999
        fmin = result.objective_value.numpy()
        status = -999
        converged = result.converged.numpy()
        params = OrderedDict((p, val) for p, val in zip(params, params_result))
        result = FitResult(params=params, edm=edm, fmin=fmin, info=info, loss=loss,
                           status=status, converged=converged,
                           minimizer=self.copy())
        return result

#  Copyright (c) 2020 zfit
from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp

from .baseminimizer import BaseMinimizer
from .fitresult import FitResult


class BFGSMinimizer(BaseMinimizer):

    def __init__(self, tolerance=1e-5, verbosity=5, name="BFGS_TFP", options=None, **minimizer_options):
        self.options = {} if options is None else options
        super().__init__(tolerance=tolerance, verbosity=verbosity, name=name, minimizer_options=minimizer_options)

    def _minimize(self, loss, params):
        minimizer_fn = tfp.optimizer.bfgs_minimize
        params = tuple(params)

        @tf.function(autograph=False, experimental_relax_shapes=True)
        def to_minimize_func(values):
            for param, value in zip(params, tf.unstack(values, axis=0)):
                param.set_value(value)
            value, gradients = loss.value_gradients(params=params)
            gradients = tf.stack(gradients)
            return value, gradients

        initial_inv_hessian_est = tf.linalg.tensor_diag([p.step_size for p in params])

        minimizer_kwargs = dict(
            initial_position=tf.stack(params),
            # tolerance=1e-4,
            f_relative_tolerance=self.tolerance,
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
                # 'message': result['message'],
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

#  Copyright (c) 2019 zfit

import tensorflow as tf


import tensorflow_probability as tfp

from .baseminimizer import BaseMinimizer


class BFGSMinimizer(BaseMinimizer):

    def __init__(self, tolerance=1e-5, verbosity=5, name="BFGS_TFP", **minimizer_options):
        super().__init__(tolerance=tolerance, verbosity=verbosity, name=name, minimizer_options=minimizer_options)

    def _minimize(self, loss, params):
        minimizer_fn = tfp.optimizer.bfgs_minimize
        params = tuple(params)

        @tf.function(autograph=True)
        def to_minimize_func(values):
            for param, value in zip(params, tf.unstack(values, axis=0)):
                param.set_value(value)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(params)
                value = loss.value()
            gradients = tape.gradient(value, params)
            return value, gradients

        result = minimizer_fn(to_minimize_func,
                              initial_position=self._extract_start_values(params),
                              tolerance=self.tolerance, parallel_iterations=1)

        # save result
        self._update_params(params, values=result)

        info = {'n_eval': result[0]['nfcn'],
                # 'n_iter': result['nit'],
                # 'grad': result['jac'],
                # 'message': result['message'],
                'original': result[0]}
        edm = result[0]['edm']
        fmin = result[0]['fval']
        status = -999
        converged = result[0]['is_valid']
        params = OrderedDict((p, res['value']) for p, res in zip(params, params_result))
        result = FitResult(params=params, edm=edm, fmin=fmin, info=info, loss=loss,
                           status=status, converged=converged,
                           minimizer=self.copy())
        return result

#  Copyright (c) 2019 zfit

from collections import OrderedDict
import copy

from scipy.optimize import SR1, BFGS
import tensorflow as tf

from zfit.minimizers.fitresult import FitResult
from .tf_external_optimizer import ScipyOptimizerInterface
from .baseminimizer import BaseMinimizer


class Scipy(BaseMinimizer):

    def __init__(self, minimizer='L-BFGS-B', tolerance=None, verbosity=5, name=None, **minimizer_options):
        if name is None:
            name = minimizer
        minimizer_options.update(method=minimizer)  # named method in ScipyOptimizerInterface
        super().__init__(tolerance=tolerance, name=name, verbosity=verbosity, minimizer_options=minimizer_options)
        # kwargs.update(hess=SR1())
        # kwargs.update(hess=BFGS())
        # kwargs.update(options={'maxiter': 3000, 'xtol': 1e-12})

    def _minimize(self, loss, params):
        loss = loss.value()
        # var_list = self.get_params()
        var_list = params

        # params_name = self._extract_param_names(var_list)
        def try_run(obj):
            if isinstance(obj, tf.Tensor):
                return self.sess.run(obj)
            else:
                return obj

        var_to_bounds = {p.name: (try_run(p.lower_limit), try_run(p.upper_limit)) for p in var_list}
        # TODO(Mayou36): inefficient for toys, rewrite ScipyOptimizerInterface?
        minimizer = ScipyOptimizerInterface(loss=loss, var_list=var_list,
                                            var_to_bounds=var_to_bounds,
                                            **self.minimizer_options)
        # self._scipy_minimizer = minimizer
        result = minimizer.minimize(session=self.sess)
        result_values = result['x']
        converged = result['success']
        status = result['status']
        info = {'n_eval': result['nfev'],
                'n_iter': result['nit'],
                'grad': result['jac'],
                'message': result['message'],
                'original': result}

        # TODO: make better
        edm = -999  # TODO: get from scipy result or how?
        fmin = result['fun']  # TODO: get from scipy result

        params = OrderedDict((p, v) for p, v in zip(var_list, result_values))

        fitresult = FitResult(params=params, edm=edm, fmin=fmin, info=info,
                              converged=converged, status=status,
                              loss=loss, minimizer=self.copy())

        return fitresult

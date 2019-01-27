from collections import OrderedDict
import copy

from scipy.optimize import SR1, BFGS
import tensorflow as tf

from zfit.minimizers.fitresult import FitResult
from .tf_external_optimizer import ScipyOptimizerInterface
from .baseminimizer import BaseMinimizer


class ScipyMinimizer(BaseMinimizer):

    def __init__(self, minimizer='L-BFGS-B', tolerance=None, name=None, **kwargs):
        if name is None:
            name = minimizer
        super().__init__(tolerance=tolerance, name=name)
        kwargs.update(method=minimizer)  # named method in ScipyOptimizerInterface
        # kwargs.update(hess=SR1())
        # kwargs.update(hess=BFGS())
        # kwargs.update(options={'maxiter': 3000, 'xtol': 1e-12})
        self._scipy_init_kwargs = kwargs

    def _minimize(self, loss, params):
        loss = loss.value()
        # var_list = self.get_parameters()
        var_list = params

        # params_name = self._extract_parameter_names(var_list)
        def try_run(obj):
            if isinstance(obj, tf.Tensor):
                return self.sess.run(obj)
            else:
                return obj

        var_to_bounds = {p.name: (try_run(p.lower_limit), try_run(p.upper_limit)) for p in var_list}
        minimizer = ScipyOptimizerInterface(loss=loss, var_list=var_list,
                                            var_to_bounds=var_to_bounds,
                                            **self._scipy_init_kwargs)
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

        # self.sess.run([assign(p['value']) for assign, p in zip(assign_params, params_result)])

        return fitresult
        # return self.get_state()

import iminuit
import numpy as np
import tensorflow as tf

from zfit.core.minimizer import BaseMinimizer


class MinuitMinimizer(BaseMinimizer):
    _DEFAULT_name = "MinuitMinimizer"

    def __init__(self, *args, **kwargs):
        self._minuit_minimizer = None
        self._error_methods['minos'] = self._minuit_minos  # before super call!
        self._error_methods['default'] = self._error_methods['minos']  # before super call!
        super().__init__(*args, **kwargs)

    def _minimize(self, params):
        loss = self.loss.eval()
        # params = self.get_parameters()
        gradients = tf.gradients(loss, params)
        updated_params = self._extract_update_op(params)
        placeholders = [param.placeholder for param in params]
        assign_params = self._extract_assign_method(params=params)

        def func(values):

            feed_dict = {p: v for p, v in zip(placeholders, values)}
            self.sess.run(updated_params, feed_dict=feed_dict)
            loss_new = tf.identity(loss)
            loss_evaluated = self.sess.run(loss_new)
            # print("Current loss:", loss_evaluated)
            # print("Current values:", values)
            return loss_evaluated

        def grad_func(values):
            feed_dict = {p: v for p, v in zip(placeholders, values)}
            self.sess.run(updated_params, feed_dict=feed_dict)
            gradients1 = tf.identity(gradients)
            gradients_values = self.sess.run(gradients1)
            return gradients_values

        # create Minuit compatible names
        error_limit_kwargs = {}
        for param in params:
            param_kwargs = {}
            param_kwargs[param.name] = self.sess.run(param.read_value())
            param_kwargs['limit_' + param.name] = self.sess.run([param.lower_limit, param.upper_limit])
            param_kwargs['error_' + param.name] = self.sess.run(param.step_size)

            error_limit_kwargs.update(param_kwargs)
        params_name = [param.name for param in params]

        minimizer = iminuit.Minuit(fcn=func, use_array_call=True,
                                   grad=grad_func,
                                   forced_parameters=params_name,
                                   **error_limit_kwargs)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(precision=self.tolerance, **self._current_error_options)
        params = [p_dict for p_dict in result[1]]
        self.sess.run([assign(p['value']) for assign, p in zip(assign_params, params)])

        edm = result[0]['edm']
        fmin = result[0]['fval']
        status = result[0]

        self.get_state(copy=False)._set_new_state(params=params, edm=edm, fmin=fmin, status=status)
        return self.get_state()

    def _minuit_minos(self, params=None, sigma=1.0):
        if params is None:
            params = self.get_parameters()
        params_name = self._extract_parameter_names(params=params)
        result = [self._minuit_minimizer.minos(var=p_name) for p_name in params_name][-1]  # returns every var
        result = {p_name: result[p_name] for p_name in params_name}
        for error_dict in result.values():
            error_dict['lower_error'] = error_dict['lower']  # TODO change value for protocol?
            error_dict['upper_error'] = error_dict['upper']  # TODO change value for protocol?
        return result

    def _hesse(self, params=None):
        params_name = self._extract_parameter_names(params=params)
        result = self._minuit_minimizer.hesse()
        result = {p_dict.pop('name'): p_dict for p_dict in result if params is None or p_dict['name'] in params_name}
        return result

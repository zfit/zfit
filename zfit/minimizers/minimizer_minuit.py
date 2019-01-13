from collections import OrderedDict
from typing import List

import iminuit
import tensorflow as tf

from zfit.minimizers.fitresult import FitResult
from ..core.parameter import Parameter
from .baseminimizer import BaseMinimizer


class MinuitMinimizer(BaseMinimizer):
    _DEFAULT_name = "MinuitMinimizer"

    def __init__(self, *args, **kwargs):
        self._minuit_minimizer = None
        super().__init__(*args, **kwargs)

    def _minimize(self, loss, params: List[Parameter]):
        loss = loss.value()
        gradients = tf.gradients(loss, params)
        assign_params = self._extract_assign_method(params=params)

        def func(values):

            # feed_dict = {p: v for p, v in zip(placeholders, value)}
            # self.sess.run(updated_params, feed_dict=feed_dict)
            for param, value in zip(params, values):
                param.load(value=value, session=self.sess)
            # loss_new = tf.identity(loss)
            loss_new = loss
            loss_evaluated = self.sess.run(loss_new)
            # print("Current loss:", loss_evaluated)
            # print("Current value:", value)
            return loss_evaluated

        def grad_func(values):
            # feed_dict = {p: v for p, v in zip(placeholders, value)}
            # self.sess.run(updated_params, feed_dict=feed_dict)
            for param, value in zip(params, values):
                param.load(value=value, session=self.sess)
            # gradients1 = tf.identity(gradients)
            gradients1 = gradients
            gradients_values = self.sess.run(gradients1)
            return gradients_values

        # create Minuit compatible names
        error_limit_kwargs = {}
        param_lower_upper_step = tuple(
            (param, param.lower_limit, param.upper_limit, param.step_size)
            for param in params)
        param_lower_upper_step = self.sess.run(param_lower_upper_step)
        for param, (value, low, up, step) in zip(params, param_lower_upper_step):
            param_kwargs = {}
            param_kwargs[param.name] = value
            param_kwargs['limit_' + param.name] = low, up
            param_kwargs['error_' + param.name] = step

            error_limit_kwargs.update(param_kwargs)
        params_name = [param.name for param in params]

        minimizer = iminuit.Minuit(fcn=func, use_array_call=True,
                                   grad=grad_func,
                                   forced_parameters=params_name,
                                   **error_limit_kwargs)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(precision=self.tolerance)
        params_result = [p_dict for p_dict in result[1]]
        self.sess.run([assign(p['value']) for assign, p in zip(assign_params, params_result)])

        edm = result[0]['edm']
        fmin = result[0]['fval']
        status = result[0]
        params = OrderedDict((p, res['value']) for p, res in zip(params, params_result))
        result = FitResult(params=params, edm=edm, fmin=fmin, status=status, loss=loss,
                           minimizer=self)  # TODO(Mayou36): should be a copy of self
        return result

from collections import OrderedDict
import copy
from typing import List

import iminuit
import texttable as tt
import tensorflow as tf

from .fitresult import FitResult
from ..util.cache import Cachable
from ..core.parameter import Parameter
from .baseminimizer import BaseMinimizer


class MinuitMinimizer(BaseMinimizer, Cachable):
    _DEFAULT_name = "MinuitMinimizer"

    def __init__(self, strategy=1, tolerance=None, verbosity=5, name=None, ncall=10000, **minuit_options):

        minimize_options = {}
        minimize_options['precision'] = minuit_options.pop('precision', None)
        minimize_options['ncall'] = ncall

        minimizer_init = {}
        minimizer_init['errordef'] = minuit_options.pop('errordef', None)
        minimizer_init['pedantic'] = minuit_options.pop('pedantic', False)
        if not strategy in range(3):
            raise ValueError("Strategy has to be 0, 1 or 2.")
        minimizer_setter = {'strategy': strategy}
        if minuit_options:
            raise ValueError("The following options are not (yet) supported: {}".format(minuit_options))
        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_init=minimizer_init,
                         minimize_options=minimize_options, minimizer_setter=minimizer_setter)
        self._minuit_minimizer = None

    def _minimize(self, loss, params: List[Parameter]):
        gradients = loss.gradients(params)
        loss_val = loss.value()
        self._check_gradients(params=params, gradients=gradients)
        load_params = self._extract_load_method(params=params)

        def func(values):
            do_print = self.verbosity > 5
            if do_print:
                table = tt.Texttable()
                table.header(['Parameter', 'Value'])
            for param, value in zip(params, values):
                param.load(value=value)
                if do_print:
                    table.add_row([param.name, value])
            if do_print:
                print(table.draw())

            loss_evaluated = self.sess.run(loss_val)
            # print("Current loss:", loss_evaluated)
            # print("Current value:", value)
            return loss_evaluated

        def grad_func(values):
            do_print = self.verbosity > 5
            if do_print:
                table = tt.Texttable()
                table.header(['Parameter', 'Gradient'])
            for param, value in zip(params, values):
                param.load(value=value)
                if do_print:
                    table.add_row([param.name, value])
            if do_print:
                print(table.draw())

            gradients_values = self.sess.run(gradients)
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

        minimizer_init = self.minimizer_init
        overlapping_kwargs = frozenset(error_limit_kwargs.keys()).intersection(minimizer_init.keys())
        if overlapping_kwargs:
            raise ValueError("The following `minimizer_init` arguments are defined internally and are invalid: "
                             "{}".format(overlapping_kwargs))
        error_limit_kwargs.update(minimizer_init)

        # if self._minuit_minimizer is None:
        minimizer = iminuit.Minuit(fcn=func, use_array_call=True,
                                   grad=grad_func,
                                   forced_parameters=params_name,
                                   print_level=self.verbosity,
                                   **error_limit_kwargs)
        minimizer_setter = self.minimizer_setter.copy()
        strategy = minimizer_setter.pop('strategy')
        minimizer.set_strategy(strategy)  # TODO(Mayou36): where to properly set strategy etc?
        assert not minimizer_setter, "minimizer_setter is not empty, bug. Please report. minimizer_setter:".format(
            minimizer_setter)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(**self.minimize_options)
        params_result = [p_dict for p_dict in result[1]]
        for load, p in zip(load_params, params_result):
            load(p['value'])

        info = {'n_eval': result[0]['nfcn'],
                # 'n_iter': result['nit'],
                # 'grad': result['jac'],
                # 'message': result['message'],
                'original': result[0]}
        edm = result[0]['edm']
        fmin = result[0]['fval']
        status = -999  # TODO: set?
        converged = result[0]['is_valid']
        params = OrderedDict((p, res['value']) for p, res in zip(params, params_result))
        result = FitResult(params=params, edm=edm, fmin=fmin, info=info, loss=loss_val,
                           status=status, converged=converged,
                           minimizer=self.copy())
        return result

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        self._minuit_minimizer = None
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer

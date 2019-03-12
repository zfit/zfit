from collections import OrderedDict
import copy
from typing import List

import iminuit
import texttable as tt
import tensorflow as tf

from zfit.core.interfaces import ZfitLoss
from .fitresult import FitResult
from ..util.cache import Cachable
from ..core.parameter import Parameter
from .baseminimizer import BaseMinimizer


class MinuitMinimizer(BaseMinimizer, Cachable):
    _DEFAULT_name = "MinuitMinimizer"

    def __init__(self, strategy=1, tolerance=None, verbosity=5, name=None, ncall=10000, **minimizer_options):

        minimizer_options['ncall'] = ncall
        if not strategy in range(3):
            raise ValueError("Strategy has to be 0, 1 or 2.")
        minimizer_options['strategy'] = strategy

        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_options=minimizer_options)
        self._minuit_minimizer = None

    def _minimize(self, loss: ZfitLoss, params: List[Parameter]):
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

        # create options
        minimizer_options = self.minimizer_options.copy()
        minimize_options = {}
        minimize_options['precision'] = minimizer_options.pop('precision', None)
        minimize_options['ncall'] = minimizer_options.pop('ncall')

        minimizer_init = {}
        if 'errordef' in minimizer_options:
            raise ValueError("errordef cannot be specified for Minuit as this is already defined in the Loss.")
        loss_errordef = loss.errordef
        if not isinstance(loss_errordef, float):
            loss_errordef = 1.0  # default of minuit
        minimizer_init['errordef'] = loss_errordef
        minimizer_init['pedantic'] = minimizer_options.pop('pedantic', False)

        minimizer_setter = {}
        minimizer_setter['strategy'] = minimizer_options.pop('strategy')
        if minimizer_options:
            raise ValueError("The following options are not (yet) supported: {}".format(minimizer_options))

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

        strategy = minimizer_setter.pop('strategy')
        minimizer.set_strategy(strategy)
        assert not minimizer_setter, "minimizer_setter is not empty, bug. Please report. minimizer_setter:".format(
            minimizer_setter)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(**minimize_options)
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
        status = -999
        converged = result[0]['is_valid']
        params = OrderedDict((p, res['value']) for p, res in zip(params, params_result))
        result = FitResult(params=params, edm=edm, fmin=fmin, info=info, loss=loss,
                           status=status, converged=converged,
                           minimizer=self.copy())
        return result

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        self._minuit_minimizer = None
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer

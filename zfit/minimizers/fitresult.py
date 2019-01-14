from collections import OrderedDict
from typing import Dict

import zfit
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..util.temporary import TemporarilySet
from ..util.container import convert_to_container
from ..util.exception import NotMinimizedError


def _hesse_minuit(result: "FitResult", params):
    fit_result = result
    params_name = [param.name for param in params]
    result = fit_result.minimizer.hesse()
    result = {p_dict.pop('name'): p_dict for p_dict in result if params is None or p_dict['name'] in params_name}
    return result


def _minos_minuit(result, params, sigma=1.0):
    fitresult = result
    params_name = [param.name for param in params]
    result = [fitresult.minimizer.minos(var=p_name) for p_name in params_name][-1]  # returns every var
    result = {p_name: result[p_name] for p_name in params_name}
    for error_dict in result.values():
        error_dict['lower_error'] = error_dict['lower']  # TODO change value for protocol?
        error_dict['upper_error'] = error_dict['upper']  # TODO change value for protocol?
    return result


class FitResult:
    _default_hesse = 'minuit'
    _hesse_methods = {'minuit': _hesse_minuit}
    _default_error = 'minuit_minos'
    _error_methods = {"minuit_minos": _minos_minuit}

    def __init__(self, params: Dict[ZfitParameter, float], edm: float, fmin: float, status: dict, loss: ZfitLoss,
                 minimizer: "ZfitMinimizer"):
        self._params = self._input_convert_params(params)
        self._edm = edm
        self._fmin = fmin
        self._status = status
        self._loss = loss
        self._minimizer = minimizer

        self._sess = None

    def _input_convert_params(self, params):
        params = OrderedDict((p, OrderedDict((('value', v),))) for p, v in params.items())
        return params

    # def get_parameters(self):  # TODO: more args?
    #     params = self._parameters
    #     if params is None:
    #         raise NotMinimizedError("No minimization performed so far. No params set.")
    #     return params
    def set_sess(self, sess):
        value = sess

        def getter():
            return self._sess  # use private attribute! self.sess creates default session

        def setter(value):
            self.sess = value

        return TemporarilySet(value=value, setter=setter, getter=getter)

    @property
    def sess(self):
        sess = self._sess
        if sess is None:
            sess = zfit.run.sess
        return sess

    @sess.setter
    def sess(self, sess):
        self._sess = sess

    @property
    def params(self):
        return self._params

    @property
    def edm(self):
        """The estimated distance to the minimum.

        Returns:
            numeric
        """
        edm = self._edm
        return edm

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def loss(self):
        return self._loss

    @property
    def fmin(self):
        """Function value at the minimum.

        Returns:
            numeric
        """
        fmin = self._fmin
        return fmin

    @property
    def status(self):
        status = self._status
        return status

    def hesse(self, params=None, method='minuit'):
        if params is not None:
            params = convert_to_container(params)
        else:
            params = list(self.params.keys())
        return self._hesse(params=params, method=method)

    def _hesse(self, params, method):
        if not callable(method):
            try:
                method = self._hesse_methods[method]
            except KeyError:
                raise KeyError("The following method is not a valid, implemented method: {}".format(method))
        return method(result=self, params=params)
    #
    # @abc.abstractmethod
    # def error(self, params=None, sigma=1., sess=None):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def set_error_method(self, method):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def set_error_options(self, replace=False, **options):
    #     raise NotImplementedError
    #
    # def _raises_error_method(*_, **__):
    #     raise NotImplementedError("No error method specified or implemented as default")

    #
    # def hesse(self, params: ztyping.ParamsOrNameType = None, sess: ztyping.SessionType = None) -> Dict:
    #     """Calculate and set for `params` the symmetric error using the Hessian matrix.
    #
    #     Args:
    #         params (list(`zfit.FitParameters` or str)): The parameters or their names to calculate the
    #             Hessian symmetric error.
    #         sess (`tf.Session` or None): A TensorFlow session to use for the calculation.
    #
    #     Returns:
    #         `dict`: A `dict` containing as keys the parameter names and as value a `dict` which
    #             contains (next to probably more things) a key 'error', holding the calculated error.
    #             Example: result['par1']['error'] -> the symmetric error on 'par1'
    #     """
    #     params = self._check_input_params(params)
    #
    #     with self.set_sess(sess=sess):
    #         errors = self._hesse(params=params)
    #         for param in params:
    #             param.error = errors[param.name]['error']
    #
    #         return errors
    # def error(self, params: ztyping.ParamsOrNameType = None, sess: ztyping.SessionType = None) -> Dict:
    #     """Calculate and set for `params` the asymmetric error using the set error method.
    #
    #     Args:
    #         params (list(`zfit.FitParameters` or str)): The parameters or their names to calculate the
    #              errors. If `params` is `None`, use all *floating* parameters.
    #         sess (`tf.Session` or None): A TensorFlow session to use for the calculation.
    #
    #     Returns:
    #         `dict`: A `dict` containing as keys the parameter names and as value a `dict` which
    #             contains (next to probably more things) two keys 'lower_error' and 'upper_error',
    #             holding the calculated errors.
    #             Example: result['par1']['upper_error'] -> the asymmetric upper error of 'par1'
    #     """
    #     params = self._check_input_params(params)
    #     with self.set_sess(sess=sess):
    #         error_method = self._current_error_method
    #         errors = error_method(params=params, **self._current_error_options)
    #         for param in params:
    #             param.lower_error = errors[param.name]['lower_error']
    #             param.upper_error = errors[param.name]['upper_error']
    #         return errors
    #
    # def set_error_method(self, method):
    #     if isinstance(method, str):
    #         try:
    #             method = self._error_methods[method]
    #         except AttributeError:
    #             raise AttributeError("The error method '{}' is not registered with the minimizer.".format(method))
    #     elif callable(method):
    #         self._current_error_method = method
    #     else:
    #         raise ValueError("Method {} is neither a valid method name nor a callable function.".format(method))
    #
    # def set_error_options(self, replace: bool = False, **options):
    #     """Specify the options for the `error` calculation.
    #
    #     Args:
    #         replace (bool): If True, replace the current options. If False, only update
    #             (add/overwrite existing).
    #         **options (keyword arguments): The keyword arguments that will be given to `error`.
    #     """
    #         self._current_error_options = {}
    #     self._current_error_options.update(options)

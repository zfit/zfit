from collections import OrderedDict

from ..util.exception import NotMinimizedError


class FitResult:

    def __init__(self, params, edm, fmin, status, loss, minimizer):
        self._parameters = params
        self._edm = edm
        self._fmin = fmin
        self._status = status
        self._loss = loss
        self._minimizer = minimizer

    def _set_parameters(self, params):  # TODO: define exactly one way to do it
        try:
            self._parameters = OrderedDict((p.name, p) for p in params)
        except AttributeError:
            self._parameters = OrderedDict((p['name'], p) for p in params)

    def get_parameters(self):  # TODO: more args?
        params = self._parameters
        if params is None:
            raise NotMinimizedError("No minimization performed so far. No params set.")
        return params

    @property
    def edm(self):
        edm = self._edm
        return

    @property
    def fmin(self):
        fmin = self._fmin
        return fmin

    @property
    def status(self):
        status = self._status
        return status

    # # just copy pasted
    # @abc.abstractmethod
    # def hesse(self, params=None, sess=None):
    #     raise NotImplementedError
    #
    # def _hesse(self, params):
    #     raise NotImplementedError
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
    # @property
    # def edm(self):
    #     """The estimated distance to the minimum.
    #
    #     Returns:
    #         numeric
    #     """
    #     return self.get_state(copy=False).edm
    #
    # @property
    # def fmin(self):
    #     """Function value at the minimum.
    #
    #     Returns:
    #         numeric
    #     """
    #     return self.get_state(copy=False).fmin
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
    #     if replace:
    #         self._current_error_options = {}
    #     self._current_error_options.update(options)

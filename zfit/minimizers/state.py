from collections import OrderedDict

from zfit.util.exception import NotMinimizedError


class MinimizerState:

    def __init__(self):
        self._parameters = None
        self._edm = None
        self._fmin = None
        self._status = None

    def _set_new_state(self, params, edm, fmin, status):
        self._set_parameters(params)
        self._set_edm(edm)
        self._set_fmin(fmin)
        self._set_status(status)

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
        if edm is None:
            raise NotMinimizedError("No minimization performed so far. No edm set.")
        return

    def _set_edm(self, edm):
        self._edm = edm

    @property
    def fmin(self):
        fmin = self._fmin
        if fmin is None:
            raise NotMinimizedError("No minimization performed so far. No fmin set.")
        return fmin

    def _set_fmin(self, fmin):
        self._fmin = fmin

    @property
    def status(self):
        status = self._status
        if status is None:
            raise NotMinimizedError("No minimization performed so far. No status set.")
        return status

    def _set_status(self, status):
        self._status = status

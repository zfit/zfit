#  Copyright (c) 2019 zfit

import abc
from abc import ABCMeta, abstractmethod


class ZfitResult:
    @abstractmethod
    def hesse(self, params, method):
        """Calculate for `params` the symmetric error using the Hessian matrix.

        Args:
            params (list(`zfit.FitParameters`)): The parameters  to calculate the
                Hessian symmetric error. If None, use all parameters.
            method (str): the method to calculate the hessian. Can be {'minuit'} or a callable.

        Returns:
            OrderedDict: Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'error': sym_error}.

                So given param_a (from zfit.Parameter(.))
                `error_a = result.hesse(params=param_a)[param_a]['error']`
                error_a is the hessian error.
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, params, method, sigma):
        """Calculate and set for `params` the asymmetric error using the set error method.

            Args:
                params (list(`zfit.FitParameters` or str)): The parameters or their names to calculate the
                     errors. If `params` is `None`, use all *floating* parameters.
                method (str or Callable): The method to use to calculate the errors. Valid choices are
                    {'minuit_minos'} or a Callable.

            Returns:
                `OrderedDict`: A `OrderedDict` containing as keys the parameter names and as value a `dict` which
                    contains (next to probably more things) two keys 'lower' and 'upper',
                    holding the calculated errors.
                    Example: result['par1']['upper'] -> the asymmetric upper error of 'par1'
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def minimizer(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fmin(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loss(self):
        raise NotImplementedError


class ZfitMinimizer:
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self, loss, params=None):
        raise NotImplementedError

    def _minimize(self, loss, params):
        raise NotImplementedError

    def _minimize_with_step(self, loss, params):
        raise NotImplementedError

    def step(self, loss, params=None):
        raise NotImplementedError

    def _step_tf(self, loss, params):
        raise NotImplementedError

    def _step(self, loss, params):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tolerance(self):
        raise NotImplementedError

    def _tolerance(self):
        raise NotImplementedError

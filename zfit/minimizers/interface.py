#  Copyright (c) 2022 zfit
from __future__ import annotations

import abc
from abc import abstractmethod


class ZfitResult:
    @abstractmethod
    def hesse(self, params, method):
        """Calculate for ``params`` the symmetric error using the Hessian matrix.

        Args:
            params: The parameters  to calculate the
                Hessian symmetric error. If None, use all parameters.
            method: the method to calculate the hessian. Can be {'minuit'} or a callable.

        Returns:
            Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'hesse': sym_error}.

                So given param_a (from zfit.Parameter(.))
                ``error_a = result.hesse(params=param_a)[param_a]['hesse']``
                error_a is the hessian error.
        """
        raise NotImplementedError

    @abstractmethod
    def errors(self, params, method, cl):
        """Calculate and set for ``params`` the asymmetric error using the set error method.

        Args:
            params: The parameters or their names to calculate the
                 errors. If ``params`` is ``None``, use all *floating* parameters.
            method: The method to use to calculate the errors. Valid choices are
                {'minuit_minos'} or a Callable.

        Returns:
            A ``OrderedDict`` containing as keys the parameter names and as value a ``dict`` which
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


class ZfitMinimizer(abc.ABC):
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self, loss, params=None, init=None):
        raise NotImplementedError

    def step(self, loss, params=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tol(self):
        raise NotImplementedError

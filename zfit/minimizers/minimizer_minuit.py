#  Copyright (c) 2021 zfit
import warnings
from typing import List, Optional

import iminuit
import numpy as np

from .baseminimizer import BaseMinimizer, minimize_supports, print_minimization_status
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import EDM
from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter, set_values
from ..settings import run
from ..util.cache import GraphCachable
from ..util.exception import MaximumIterationReached


class Minuit(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "Minuit"

    def __init__(self, strategy: ZfitStrategy = None, minimize_strategy: int = None, tol: float = None,
                 verbosity: int = 5, name: str = None,
                 ncall: Optional[int] = None, minuit_grad: Optional[bool] = None, use_minuit_grad: bool = None,
                 minimizer_options=None):
        """Minuit is a longstanding and well proven algorithm of the L-BFGS-B class implemented in
        `iminuit<https://iminuit.readthedocs.io/en/stable/>`_.

        The package iminuit is the fast, interactive minimizer based on the Minuit2 C++ library maintained
        by CERN’s ROOT team.

        Args:
            strategy: A :py:class:`~zfit.minimizer.baseminimizer.ZfitStrategy` object that defines the behavior of
            the minimizer in certain situations.
            minimize_strategy: A number used by minuit to define the strategy, either 0, 1 or 2.
            tol: Stopping criteria: the Estimated Distance to Minimum (EDM) has to be lower than `tolerance`
            verbosity: Regulates how much will be printed during minimization. Values between 0 and 10 are valid.
            name: Name of the minimizer
            ncall: Maximum number of minimization steps.
            minuit_grad: If True, iminuit uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
            use_minuit_grad: If True, iminuit uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
        """

        self._internal_maxiter = 20
        minuit_grad = use_minuit_grad if use_minuit_grad is not None else minuit_grad
        minimizer_options = {} if minimizer_options is None else minimizer_options
        minimizer_options['ncall'] = 0 if ncall is None else ncall
        if minimize_strategy is None:
            minimize_strategy = 0 if not minuit_grad else 1
        else:
            minimize_strategy = minimize_strategy
        if minimize_strategy not in range(3):
            raise ValueError(f"minimize_strategy has to be 0, 1 or 2, not {minimize_strategy}.")
        minimizer_options['strategy'] = minimize_strategy

        super().__init__(name=name, strategy=strategy, tol=tol, verbosity=verbosity, criterion=None,
                         maxiter=1e20,
                         minimizer_options=minimizer_options)
        minuit_grad = True if minuit_grad is None else minuit_grad
        self._minuit_minimizer = None
        self._use_tfgrad_internal = not minuit_grad
        self.minuit_grad = minuit_grad

    # TODO 0.7: legacy, remove `_use_tfgrad`
    @property
    def _use_tfgrad(self):
        warnings.warn("Do not use `minimizer._use_tfgrad`, this will be removed. Use `minuit_grad` instead in the"
                      " initialization.", stacklevel=2)
        return self._use_tfgrad_internal

    @minimize_supports()
    def _minimize(self, loss: ZfitLoss, params: List[Parameter], init):
        if init:
            set_values(params=params, values=init)
        criterion = self.create_criterion(loss, params)

        minimizer, minimize_options, evaluator = self._make_minuit(loss, params, init)

        self._minuit_minimizer = minimizer

        valid = True
        message = ""
        maxiter_reached = False
        for i in range(self._internal_maxiter):

            # perform minimization
            try:
                minimizer = minimizer.migrad(**minimize_options)
            except MaximumIterationReached as error:
                if minimizer is None:  # it didn't even run once
                    raise MaximumIterationReached("Maximum iteration reached on first wrapped minimizer call. This"
                                                  "is likely to a too low number of maximum iterations (currently"
                                                  f" {evaluator.maxiter}) or wrong internal tolerances, in which"
                                                  f" case: please fill an issue on github.") from error
                maxiter_reached = True
                message = "Maxiter reached"
            else:
                if evaluator.maxiter is not None:
                    maxiter_reached = evaluator.niter > evaluator.maxiter
            if type(criterion) == EDM:  # use iminuits edm
                criterion.last_value = minimizer.fmin.edm
                converged = not minimizer.fmin.is_above_max_edm
            else:
                fitresult = FitResult.from_minuit(loss=loss, params=params, minuit=minimizer, minimizer=self,
                                                  valid=valid, message=message)
                converged = criterion.converged(fitresult)

            if self.verbosity > 5:
                internal_tol = {'edm_minuit': minimizer.fmin.edm}

                print_minimization_status(converged=converged,
                                          criterion=criterion,
                                          evaluator=evaluator,
                                          i=i,
                                          fmin=minimizer.fval,
                                          internal_tol=internal_tol)

            if converged or maxiter_reached:
                set_values(params, minimizer.values)  # make sure it's at the right value
                break

        fitresult = FitResult.from_minuit(loss=loss,
                                          params=params,
                                          criterion=criterion,
                                          minuit=minimizer,
                                          minimizer=self.copy(),
                                          valid=valid,
                                          message=message)
        return fitresult

    def _make_minuit(self, loss, params, init):
        previous_result = init

        evaluator = self.create_evaluator(loss, params)
        # create options
        minimizer_options = self.minimizer_options.copy()
        minimize_options = {}
        precision = minimizer_options.pop('precision', None)
        minimize_options['ncall'] = minimizer_options.pop('ncall')
        minimizer_init = {}
        if 'errordef' in minimizer_options:
            raise ValueError("errordef cannot be specified for Minuit as this is already defined in the Loss.")
        loss_errordef = loss.errordef
        if not isinstance(loss_errordef, (float, int)):
            raise ValueError("errordef has to be a float")
        minimizer_init['errordef'] = loss_errordef
        minimizer_init['pedantic'] = minimizer_options.pop('pedantic', False)
        minimizer_setter = {}
        minimizer_setter['strategy'] = minimizer_options.pop('strategy')
        if self.verbosity > 6:
            minuit_verbosity = 3
        elif self.verbosity > 2:
            minuit_verbosity = 1
        else:
            minuit_verbosity = 0
        if minimizer_options:
            raise ValueError("The following options are not (yet) supported: {}".format(minimizer_options))
        init_values = np.array(run(params))
        # create Minuit compatible names
        params_name = [param.name for param in params]
        # TODO 0.7: legacy, remove `_use_tfgrad`
        grad_func = evaluator.gradient if self._use_tfgrad_internal or not self.minuit_grad else None
        minimizer = iminuit.Minuit(evaluator.value, init_values,
                                   grad=grad_func,
                                   name=params_name,
                                   )
        minimizer.precision = precision
        approx_step_sizes = {}
        # get possible initial step size from previous minimizer
        if previous_result:
            approx_step_sizes = previous_result.hesse(params=params, method='approx')
        empty_dict = {}
        for param in params:
            step_size = approx_step_sizes.get(param, empty_dict).get('error')
            if step_size is None and param.has_step_size:
                step_size = param.step_size
            if step_size is not None:
                minimizer.errors[param.name] = step_size
        # set limits
        for param in params:
            if param.has_limits:
                minimizer.limits[param.name] = (param.lower, param.upper)
        # set options
        minimizer.errordef = loss.errordef
        minimizer.print_level = minuit_verbosity
        strategy = minimizer_setter.pop('strategy')
        minimizer.strategy = strategy
        minimizer.tol = (self.tol / 0.002  # iminuit multiplies by default with 0.002
                         / loss.errordef)  # to account for the loss
        assert not minimizer_setter, "minimizer_setter is not empty, bug. Please report. minimizer_setter: {}".format(
            minimizer_setter)
        return minimizer, minimize_options, evaluator

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer

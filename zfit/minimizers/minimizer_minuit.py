#  Copyright (c) 2021 zfit
import warnings
from typing import List, Optional

import iminuit
import numpy as np

from .baseminimizer import BaseMinimizer, minimize_supports
from .fitresult import FitResult
from .strategy import ZfitStrategy
from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter
from ..settings import run
from ..util.cache import GraphCachable


class Minuit(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "Minuit"

    def __init__(self, strategy: ZfitStrategy = None, minimize_strategy: int = 1, tolerance: float = None,
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
            tolerance: Stopping criteria: the Estimated Distance to Minimum (EDM) has to be lower then `tolerance`
            verbosity: Regulates how much will be printed during minimization. Values between 0 and 10 are valid.
            name: Name of the minimizer
            ncall: Maximum number of minimization steps.
            minuit_grad: If True, iminuit uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
            use_minuit_grad: If True, iminuit uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
        """
        minuit_grad = use_minuit_grad if use_minuit_grad is not None else minuit_grad
        minimizer_options = {} if minimizer_options is None else minimizer_options
        minimizer_options['ncall'] = 0 if ncall is None else ncall
        if minimize_strategy not in range(3):
            raise ValueError(f"minimize_strategy has to be 0, 1 or 2, not {minimize_strategy}.")
        minimizer_options['strategy'] = minimize_strategy

        super().__init__(name=name, strategy=strategy, tolerance=tolerance, verbosity=verbosity,
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
        grad_func = evaluator.gradient if self._use_tfgrad or not self.minuit_grad else None

        minimizer = iminuit.Minuit(evaluator.value, init_values,
                                   grad=grad_func,
                                   name=params_name,
                                   )
        minimizer.precision = precision
        approx_step_sizes = previous_result.hesse(params=params, method='approx')
        for param in params:
            step_size = approx_step_sizes.get(param)
            if step_size is None and param.has_step_size:
                step_size = param.step_size
            if step_size is not None:
                minimizer.errors[param.name] = step_size

            if param.has_limits:
                minimizer.limits[param.name] = (param.lower, param.upper)
        minimizer.errordef = loss.errordef
        minimizer.print_level = minuit_verbosity
        strategy = minimizer_setter.pop('strategy')
        minimizer.strategy = strategy
        minimizer.tol = self.tolerance / 1e-3  # iminuit 1e-3 and tolerance 0.1
        assert not minimizer_setter, "minimizer_setter is not empty, bug. Please report. minimizer_setter: {}".format(
            minimizer_setter)
        self._minuit_minimizer = minimizer
        result = minimizer.migrad(**minimize_options)
        fitresult = FitResult.from_minuit(loss=loss,
                                          params=params,
                                          result=result,
                                          minimizer=self.copy())
        return fitresult

    def copy(self):
        tmp_minimizer = self._minuit_minimizer
        new_minimizer = super().copy()
        new_minimizer._minuit_minimizer = tmp_minimizer
        return new_minimizer

#  Copyright (c) 2021 zfit
from typing import Optional, Union, Callable

from .baseminimizer import NOT_SUPPORTED
from .minimizers_scipy import ScipyBaseMinimizer
from .strategy import ZfitStrategy
from .termination import ConvergenceCriterion
# import cyipopt

class IPopt(ScipyBaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 # maxcor: Optional[int] = None,
                 # maxls: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name="IPopt"):
        options = {}

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'ftol': None, 'gtol': None}

        super().__init__(method=None, internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         # minimize_func=cyipopt.scipy_interface.minimize_ipopt,
                         strategy=strategy, criterion=criterion, name=name)

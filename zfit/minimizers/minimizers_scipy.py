#  Copyright (c) 2021 zfit
import copy
import inspect
import math
from typing import Callable, Mapping, Optional, Union

import numpy as np
import scipy.optimize
from scipy.optimize import SR1

from ..core.parameter import set_values
from ..settings import run
from ..util.exception import MaximumIterationReached
from ..util.warnings import warn_experimental_feature
from .baseminimizer import (NOT_SUPPORTED, BaseMinimizer, minimize_supports,
                            print_minimization_status)
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import CRITERION_NOT_AVAILABLE, ConvergenceCriterion


class ScipyBaseMinimizer(BaseMinimizer):
    _ALLOWED_GRADIENT = {None, True, False, 'zfit'}
    _ALLOWED_HESSIAN = {None, True, False, 'zfit'}

    def __init__(self,
                 method: str,
                 tol: Optional[float],
                 internal_tol: Mapping[str, Optional[float]],
                 gradient: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy, NOT_SUPPORTED]],
                 maxiter: Optional[Union[int, str]] = None,
                 minimizer_options: Optional[Mapping[str, object]] = None,
                 verbosity: Optional[int] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 minimize_func: Optional[callable] = None,
                 name: str = "ScipyMinimizer"
                 ) -> None:
        """Base minimizer wrapping the SciPy librarys optimize module.

        Args:
            method: Name of the method as given to :func:~`scipy.optimize.minimize`
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the `value`, 'gradient` or `hessian`. |@docend:minimizer.maxiter|
            minimizer_options:
            verbosity: |@doc:minimizer.verbosity| Verbosity of the minimizer.
                A value above 5 starts printing more
                output with a value of 10 printing every
                evaluation of the loss function and gradient. |@docend:minimizer.verbosity|
            strategy: |@doc:minimizer.strategy| Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs in which case |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            minimize_func:
            name: |@doc:minimizer.name| Human readable name of the minimizer. |@docend:minimizer.name|
        """
        self._minimize_func = scipy.optimize.minimize if minimize_func is None else minimize_func

        minimizer_options = {} if minimizer_options is None else minimizer_options
        minimizer_options = copy.copy(minimizer_options)
        minimizer_options['method'] = method

        if 'options' not in minimizer_options:
            minimizer_options['options'] = {}

        if gradient is not NOT_SUPPORTED:
            if gradient is False or gradient is None:
                gradient = 'zfit'

                # raise ValueError("grad cannot be False for SciPy minimizer.")
            elif gradient is True:
                gradient = None
            minimizer_options['grad'] = gradient
        if hessian is not NOT_SUPPORTED:
            if isinstance(hessian, scipy.optimize.HessianUpdateStrategy) and not inspect.isclass(hessian):
                raise ValueError("If `hesse` is a HessianUpdateStrategy, it has to be a class that takes `init_scale`,"
                                 " not an instance. For further modification of other initial parameters, make a"
                                 " subclass of the update strategy.")
            if hessian is True:
                hessian = None
            elif hessian is False or gradient is None:
                hessian = 'zfit'

            minimizer_options['hess'] = hessian
        self._internal_tol = internal_tol
        self._internal_maxiter = 20
        super().__init__(name=name, tol=tol, verbosity=verbosity, minimizer_options=minimizer_options,
                         strategy=strategy, criterion=criterion, maxiter=maxiter)

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init: FitResult):
        if init:
            set_values(params=params, values=init)

        evaluator = self.create_evaluator(loss=loss, params=params)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        minimizer_options = self.minimizer_options.copy()

        minimizer_options['bounds'] = limits

        use_gradient = 'grad' in minimizer_options
        if use_gradient:
            gradient = minimizer_options.pop('grad')
            gradient = evaluator.gradient if gradient == 'zfit' else gradient
            minimizer_options['jac'] = gradient

        use_hessian = 'hess' in minimizer_options
        if use_hessian:
            hessian = minimizer_options.pop('hess')
            hessian = evaluator.hessian if hessian == 'zfit' else hessian
            minimizer_options['hess'] = hessian

            is_update_strat = inspect.isclass(hessian) and issubclass(hessian,
                                                                      scipy.optimize.HessianUpdateStrategy)

            init_scale = 'auto'
            # get possible initial step size from previous minimizer
            if init:
                approx_step_sizes = init.hesse(params=params, method='approx')
            else:
                approx_step_sizes = None

        maxiter = self.get_maxiter(len(params))
        if maxiter is not None:
            # stop 3 iterations earlier than we
            minimizer_options['options']['maxiter'] = maxiter - 3 if maxiter > 10 else maxiter

        minimizer_options['options']['disp'] = self.verbosity > 6

        # tolerances and criterion
        criterion = self.create_criterion(loss, params)

        init_tol = min([math.sqrt(loss.errordef * self.tol), loss.errordef * self.tol * 1e3])
        internal_tol = self._internal_tol
        internal_tol = {tol: init_tol if init is None else init for tol, init in internal_tol.items()}

        valid = None
        message = None
        optimize_results = None
        for i in range(self._internal_maxiter):

            # update from previous run/result
            if use_hessian and is_update_strat:
                if not isinstance(init_scale, str):
                    init_scale = np.mean(approx_step_sizes)
                minimizer_options['hess'] = hessian(init_scale=init_scale)

            for tol, val in internal_tol.items():
                minimizer_options['options'][tol] = val

            # perform minimization
            try:
                optim_result = self._minimize_func(fun=evaluator.value, x0=init_values, **minimizer_options)
            except MaximumIterationReached as error:
                if optim_result is None:  # it didn't even run once
                    raise MaximumIterationReached("Maximum iteration reached on first wrapped minimizer call. This"
                                                  "is likely to a too low number of maximum iterations (currently"
                                                  f" {evaluator.maxiter}) or wrong internal tolerances, in which"
                                                  f" case: please fill an issue on github.") from error
                maxiter_reached = True
                valid = False
                message = "Maxiter reached, terminated without convergence"
            else:
                maxiter_reached = evaluator.niter > evaluator.maxiter

            values = optim_result['x']

            fmin = optim_result.fun
            set_values(params, values)

            optimize_results = combine_optimize_results(
                [optim_result] if optimize_results is None else [optimize_results, optim_result])
            result_prelim = FitResult.from_scipy(loss=loss,
                                                 params=params,
                                                 result=optimize_results,
                                                 minimizer=self,
                                                 edm=CRITERION_NOT_AVAILABLE,
                                                 criterion=None,
                                                 message='INTERNAL for Criterion',
                                                 valid=valid)

            if use_hessian:
                approx_step_sizes = result_prelim.hesse(params=params, method='approx')
            converged = criterion.converged(result_prelim)
            valid = converged
            edm = criterion.last_value

            if self.verbosity > 5:
                print_minimization_status(converged=converged,
                                          criterion=criterion,
                                          evaluator=evaluator,
                                          i=i,
                                          fmin=fmin,
                                          internal_tol=internal_tol)

            if converged or maxiter_reached:
                break
            init_values = values

            # update the tolerances
            self._update_tol_inplace(criterion_value=edm, internal_tol=internal_tol)

        else:
            message = f"Invalid, criterion {criterion.name} is {edm}, target {self.tol} not reached."
            valid = False
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=optimize_results,
            minimizer=self,
            valid=valid,
            criterion=criterion,
            edm=edm,
            message=message,
            niter=evaluator.niter,
            evaluator=evaluator,
        )


class ScipyLBFGSBV1(ScipyBaseMinimizer):
    _ALLOWED_GRADIENT = ScipyBaseMinimizer._ALLOWED_GRADIENT.union(['2-point', '3-point',
                                                                    # 'cs'  # works badly
                                                                    ])
    _ALLOWED_HESSIAN = ScipyBaseMinimizer._ALLOWED_HESSIAN.union(['2-point', '3-point', 'cs',
                                                                  scipy.optimize.BFGS, scipy.optimize.SR1])

    def __init__(self,
                 tol: Optional[float] = None,
                 maxcor: Optional[int] = None,
                 maxls: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy L-BFGS-B V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            maxcor: |@doc:minimizer.maxcor||@docend:minimizer.maxcor|
            maxls: |@doc:minimizer.init.maxls||@docend:minimizer.init.maxls|
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            gradient: |@doc:minimizer.scipy.gradient||@docend:minimizer.scipy.gradient|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if maxcor is not None:
            options['maxcor'] = maxcor
        if maxls is not None:
            options['maxls'] = maxls

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'ftol': None, 'gtol': None}

        super().__init__(method="L-BFGS-B", internal_tol=scipy_tols, gradient=gradient,
                         hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustKrylovV1(ScipyBaseMinimizer):
    _ALLOWED_GRADIENT = ScipyBaseMinimizer._ALLOWED_GRADIENT.union(['2-point', '3-point',
                                                                    # 'cs'  # works badly
                                                                    ])
    _ALLOWED_HESSIAN = ScipyBaseMinimizer._ALLOWED_HESSIAN.union(['2-point', '3-point',
                                                                  # 'cs',
                                                                  scipy.optimize.BFGS, scipy.optimize.SR1])

    def __init__(self,
                 tol: Optional[float] = None,
                 inexact: Optional[bool] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy trust-krylov V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            inexact:
            gradient:
            hessian:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if inexact is not None:
            options['inexact'] = inexact

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'gtol': None}

        super().__init__(method="trust-constr", internal_tol=scipy_tols, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustNCGV1(ScipyBaseMinimizer):
    def __init__(self,
                 tol: Optional[float] = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy trust-ncg V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            eta:
            max_trust_radius:
            gradient:
            hessian:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if eta is not None:
            options['eta'] = eta
        if max_trust_radius is not None:
            options['max_trust_radius'] = max_trust_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'gtol': None}

        super().__init__(method="trust-constr", internal_tol=scipy_tols, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustConstrV1(ScipyBaseMinimizer):
    def __init__(self,
                 tol: Optional[float] = None,
                 init_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy trust-constr V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            init_trust_radius:
            gradient:
            hessian:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if init_trust_radius is not None:
            options['initial_tr_radius'] = init_trust_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'gtol': None, 'xtol': None}

        super().__init__(method="trust-constr", internal_tol=scipy_tols, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyNewtonCGV1(ScipyBaseMinimizer):

    @warn_experimental_feature
    def __init__(self,
                 tol: Optional[float] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = 'zfit',
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy Newton-CG V1"
                 ) -> object:
        """WARNING! This algorithm seems unstable and may does not perform well!

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            gradient:
            hessian:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'xtol': None}

        method = "Newton-CG"
        super().__init__(method=method, internal_tol=scipy_tols, gradient=gradient, hessian=hessian,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTruncNCV1(ScipyBaseMinimizer):
    def __init__(self, tol: Optional[float] = None,
                 maxiter_cg: Optional[int] = None,  # maxCGit
                 maxstep_ls: Optional[int] = None,  # stepmx
                 eta: Optional[float] = None,
                 rescale: Optional[float] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy Truncated Newton Conjugate V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            maxiter_cg:
            maxstep_ls:
            eta:
            rescale:
            gradient:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if maxiter_cg is not None:
            options['maxiter_cg'] = maxiter_cg
        if eta is not None:
            options['eta'] = eta
        if maxstep_ls is not None:
            options['maxstep_ls'] = maxstep_ls
        if rescale is not None:
            options['rescale'] = rescale

        options['maxfun'] = None  # in order to use maxiter
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'xtol': None, 'ftol': None, 'gtol': None}

        method = "TNC"
        super().__init__(method=method, tol=tol, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=NOT_SUPPORTED,
                         criterion=criterion, internal_tol=scipy_tols,
                         maxiter=maxiter,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyDoglegV1(ScipyBaseMinimizer):
    def __init__(self, tol: Optional[float] = None,
                 init_trust_radius: Optional[int] = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = 'zfit',
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy Dogleg V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            init_trust_radius:
            eta:
            max_trust_radius:
            gradient:
            hessian:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        if init_trust_radius is not None:
            options['initial_tr_radius'] = init_trust_radius
        if eta is not None:
            options['eta'] = eta
        if max_trust_radius is not None:
            options['max_trust_radius'] = max_trust_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'gtol': None}

        super().__init__(method="trust-constr", internal_tol=scipy_tols, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyPowellV1(ScipyBaseMinimizer):
    def __init__(self,
                 tol: Optional[float] = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy Powell V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'xtol': None, 'ftol': None}

        method = "Powell"
        super().__init__(method=method, internal_tol=scipy_tols, gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED, minimizer_options=minimizer_options, tol=tol,
                         maxiter=maxiter,
                         verbosity=verbosity, strategy=strategy, criterion=criterion, name=name)


class ScipySLSQPV1(ScipyBaseMinimizer):
    def __init__(self,
                 tol: Optional[float] = None,
                 gradient: Optional[Union[Callable, str]] = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy SLSQP V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            gradient:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        if gradient is None:
            gradient = 'zfit'
        options = {}
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'ftol': None}

        method = "SLSQP"
        super().__init__(method=method, internal_tol=scipy_tols, gradient=gradient, hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyNelderMeadV1(ScipyBaseMinimizer):
    def __init__(self,
                 tol: Optional[float] = None,
                 adaptive: Optional[bool] = True,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name: str = "SciPy Nelder-Mead V1"
                 ) -> None:
        """

        Args:
            tol: |@doc:minimizer.tol||@docend:minimizer.tol|
            adaptive:
            verbosity: |@doc:minimizer.verbosity||@docend:minimizer.verbosity|
            maxiter: |@doc:minimizer.maxiter||@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion||@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy||@docend:minimizer.strategy|
            name: |@doc:minimizer.name||@docend:minimizer.name|
        """
        options = {}
        minimizer_options = {}

        if adaptive is not None:
            options['adaptive'] = adaptive
        if options:
            minimizer_options['options'] = options

        scipy_tols = {'fatol': None, 'xatol': None}

        method = "Nelder-Mead"
        super().__init__(method=method, internal_tol=scipy_tols, gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED, minimizer_options=minimizer_options, tol=tol,
                         maxiter=maxiter,
                         verbosity=verbosity, strategy=strategy, criterion=criterion, name=name)


def combine_optimize_results(results):
    if len(results) == 1:
        return results[0]
    result = results[-1]
    for field in ['nfev', 'njev', 'nhev', 'nit']:
        if field in result:
            result[field] = sum(res[field] for res in results)
    return result

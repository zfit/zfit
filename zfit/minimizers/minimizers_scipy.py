#  Copyright (c) 2021 zfit
import copy
import inspect
import math
import warnings
from typing import Optional, Dict, Callable, Union, Mapping

import numpy as np
import scipy.optimize
from scipy.optimize import SR1, LbfgsInvHessProduct

from .baseminimizer import BaseMinimizer, minimize_supports, NOT_SUPPORTED, print_minimization_status
from .evaluation import LossEval
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import ConvergenceCriterion, CRITERION_NOT_AVAILABLE, EDM
from ..core.parameter import set_values
from ..settings import run
from ..util.exception import MaximumIterationReached


class ScipyBaseMinimizer(BaseMinimizer):
    def __init__(self,
                 method: str,
                 internal_tolerances: Mapping[str, Optional[float]],
                 gradient: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy, NOT_SUPPORTED]],
                 maxiter: Optional[Union[int, str]],
                 minimizer_options: Mapping[str, object],
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 minimize_func: Optional[callable] = None,
                 name="ScipyMinimizer"):
        minimize_func = scipy.optimize.minimize if minimize_func is None else minimize_func
        self._minimize_func = minimize_func
        minimizer_options = copy.copy(minimizer_options)
        minimizer_options['method'] = method
        if 'options' not in minimizer_options:
            minimizer_options['options'] = {}

        if gradient is not NOT_SUPPORTED:
            if gradient is False:
                raise ValueError("grad cannot be False for SciPy minimizer.")
            minimizer_options['grad'] = gradient
        if hessian is not NOT_SUPPORTED:
            if isinstance(hessian, scipy.optimize.HessianUpdateStrategy) and not inspect.isclass(hessian):
                raise ValueError("If `hesse` is a HessianUpdateStrategy, it has to be a class that takes `init_scale`,"
                                 " not an instance. For further modification of other initial parameters, make a"
                                 " subclass of the update strategy.")
            minimizer_options['hess'] = hessian
        self._internal_tolerances = internal_tolerances
        self._internal_maxiter = 20
        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_options=minimizer_options,
                         strategy=strategy, criterion=criterion, maxiter=maxiter)

    @minimize_supports()
    def _minimize(self, loss, params, init: FitResult):
        previous_result = init
        evaluator = self.create_evaluator(loss, params)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        minimizer_options = self.minimizer_options.copy()
        minimizer_options['bounds'] = limits

        use_gradient = 'grad' in minimizer_options
        if use_gradient:
            gradient = minimizer_options.pop('grad')
            gradient = evaluator.gradient if gradient == 'zfit' else gradient
            minimizer_options['jac'] = gradient

        inv_hessian = None
        use_hessian = 'hess' in minimizer_options
        if use_hessian:
            hessian = minimizer_options.pop('hess')
            hessian = evaluator.hessian if hessian == 'zfit' else hessian
            minimizer_options['hess'] = hessian

            is_update_strat = inspect.isclass(hessian) and issubclass(hessian,
                                                                      scipy.optimize.HessianUpdateStrategy)

            init_scale = 'auto'
            if previous_result:
                inv_hessian = previous_result.approx.inv_hessian(params)

        maxiter = self.get_maxiter(len(params))
        if maxiter is not None:
            # stop 3 iterations earlier than we
            minimizer_options['options']['maxiter'] = maxiter - 3 if maxiter > 10 else maxiter

        minimizer_options['options']['disp'] = self.verbosity > 6

        # tolerances and criterion
        criterion = self.create_criterion(loss, params)

        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        internal_tol = self._internal_tolerances
        internal_tol = {tol: init_tol if init is None else init for tol, init in internal_tol.items()}

        valid = None
        valid_message = None
        optimize_results = None
        for i in range(self._internal_maxiter):

            # update from previous run/result
            if use_hessian and is_update_strat:
                if not isinstance(init_scale, str):
                    init_scale = np.mean(init_scale)
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
                valid_message = "Maxiter reached, terminated without convergence"
            else:
                maxiter_reached = evaluator.niter > evaluator.maxiter

            xvalues = optim_result['x']

            fmin = optim_result.fun
            set_values(params, xvalues)

            optimize_results = combine_optimize_results(
                [optim_result] if optimize_results is None else [optimize_results, optim_result])
            result_prelim = FitResult.from_scipy(loss=loss, params=params, result=optimize_results, minimizer=self,
                                                 edm=CRITERION_NOT_AVAILABLE, valid=valid)

            if use_hessian:
                inv_hessian = optim_result.approx.get('hess_inv')
            converged = criterion.converged(result_prelim)
            criterion_value = criterion.last_value
            if isinstance(criterion, EDM):
                edm = criterion.last_value
            else:
                edm = CRITERION_NOT_AVAILABLE

            if self.verbosity > 5:
                print_minimization_status(converged=converged,
                                          criterion=criterion,
                                          evaluator=evaluator,
                                          i=i,
                                          fmin=fmin,
                                          internal_tol=internal_tol)

            if converged or maxiter_reached:
                break
            if use_hessian and inv_hessian is not None:
                init_scale = inv_hessian
            init_values = xvalues

            # update the tolerances
            self._update_tol_inplace(criterion_value=criterion_value, internal_tol=internal_tol)

        else:
            valid = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tolerance} not reached."
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=optimize_results,
            minimizer=self,
            valid=valid,
            criterion=criterion,
            edm=edm,
            message=valid_message,
            niter=evaluator.niter,
        )


class Scipy(BaseMinimizer):

    def __init__(self, algorithm: str = 'L-BFGS-B', tolerance: Optional[float] = None,
                 strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5,
                 name: Optional[str] = None, scipy_grad: Optional[bool] = None,
                 # criterion: ConvergenceCriterion = None,
                 minimizer_options: Optional[Dict[str, object]] = None, minimizer=None):
        """SciPy optimizer algorithms.

        This is a wrapper for all the SciPy optimizers. More information can be found in their docs.

        Args:
            algorithm: Name of the minimization algorithm to use.
            tolerance: Stopping criterion of the algorithm to determine when the minimum
                has been found. The default is 1e-4, which is *different* from others.
            verbosity:             name: Name of the minimizer
            scipy_grad: If True, SciPy uses it's internal numerical gradient calculation instead of the
                (analytic/numerical) gradient provided by TensorFlow/zfit.
            name: Name of the minimizer
            minimizer_options:
        """
        # if criterion is None:
        #     criterion = EDM
        # self.criterion = criterion
        if minimizer is not None:
            warnings.warn(DeprecationWarning, "`minimizer` keyword is gone, use `algorithm`")
            algorithm = minimizer
        scipy_grad = True if scipy_grad is None else scipy_grad
        minimizer_options = {} if minimizer_options is None else minimizer_options
        if tolerance is None:
            tolerance = 1e-4
        if name is None:
            name = algorithm
        self.scipy_grad = scipy_grad
        minimizer_options = minimizer_options.copy()
        minimizer_options.update(method=algorithm)
        super().__init__(tolerance=tolerance, name=name, verbosity=verbosity,
                         strategy=strategy,
                         minimizer_options=minimizer_options)

    @minimize_supports()
    def _minimize(self, loss, params):
        # criterion = self.criterion(tolerance=self.tolerance, loss=loss)
        minimizer_options = self.minimizer_options.copy()
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        initial_values = np.array(run(params))
        limits = [(run(p.lower), run(p.upper)) for p in params]
        minimize_kwargs = {
            'jac': not self.scipy_grad,
            # TODO: properly do hessian or jac
            # 'hess': scipy.optimize.BFGS() if self.scipy_grad else None,
            # 'callback': step_callback,
            'method': minimizer_options.pop('method'),
            # 'constraints': constraints,
            'bounds': limits,
            'tol': self.tolerance,
            'options': minimizer_options.pop('options', None),
        }
        minimize_kwargs.update(self.minimizer_options)
        result = scipy.optimize.minimize(fun=evaluator.value if self.scipy_grad else evaluator.value_gradients,
                                         x0=initial_values, **minimize_kwargs)

        xvalues = result['x']
        set_values(params, xvalues)
        from .fitresult import FitResult
        return FitResult.from_scipy(loss=loss,
                                    params=params,
                                    result=result,
                                    minimizer=self
                                    )


class ScipyLBFGSBV1(ScipyBaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: Optional[int] = None,
                 maxls: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 name="Scipy L-BFGS-B V1"):
        options = {}
        if maxcor is not None:
            options['maxcor'] = maxcor
        if maxls is not None:
            options['maxls'] = maxls

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'ftol': None, 'gtol': None}

        super().__init__(method="L-BFGS-B", internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustKrylovV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 inexact: Optional[bool] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy trust-krylov V1"):
        options = {}
        if inexact is not None:
            options['inexact'] = inexact

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'gtol': None}

        super().__init__(method="trust-constr", internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustNCGV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name: str = "Scipy trust-ncg V1"):
        options = {}
        if eta is not None:
            options['eta'] = eta
        if max_trust_radius is not None:
            options['max_trust_radius'] = max_trust_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'gtol': None}

        super().__init__(method="trust-constr", internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTrustConstrV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 initial_tr_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy trust-constr V1"):
        options = {}
        if initial_tr_radius is not None:
            options['initial_tr_radius'] = initial_tr_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'gtol': None, 'xtol': None}

        super().__init__(method="trust-constr", internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyNewtonCGV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy Newton-CG V1"):
        options = {}

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'xtol': None}

        method = "Newton-CG"
        super().__init__(method=method, internal_tolerances=scipy_tolerances, gradient=gradient, hessian=hessian,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyTruncNCV1(ScipyBaseMinimizer):
    def __init__(self, tolerance: float = None,
                 maxiter_cg: Optional[int] = None,  # maxCGit
                 maxstep_ls: Optional[int] = None,  # stepmx
                 eta: Optional[float] = None,
                 rescale: Optional[float] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy Truncated Newton Conjugate V1"):
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

        scipy_tolerances = {'xtol': None, 'ftol': None, 'gtol': None}

        method = "TNC"
        super().__init__(method=method, tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=NOT_SUPPORTED,
                         criterion=criterion, internal_tolerances=scipy_tolerances,
                         maxiter=maxiter,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyDoglegV1(ScipyBaseMinimizer):
    def __init__(self, tolerance: float = None,
                 initial_trust_radius: Optional[int] = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy Dogleg V1"):
        options = {}
        if initial_trust_radius is not None:
            options['initial_tr_radius'] = initial_trust_radius
        if eta is not None:
            options['eta'] = eta
        if max_trust_radius is not None:
            options['max_trust_radius'] = max_trust_radius

        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'gtol': None}

        super().__init__(method="trust-constr", internal_tolerances=scipy_tolerances, gradient=gradient,
                         hessian=hessian,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyPowellV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy Powell V1"):
        options = {}
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'xtol': None, 'ftol': None}

        method = "Powell"
        super().__init__(method=method, internal_tolerances=scipy_tolerances, gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED, minimizer_options=minimizer_options, tolerance=tolerance,
                         maxiter=maxiter,
                         verbosity=verbosity, strategy=strategy, criterion=criterion, name=name)


class ScipySLSQPV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy SLSQP V1"):
        options = {}
        minimizer_options = {}
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'ftol': None}

        method = "SLSQP"
        super().__init__(method=method, internal_tolerances=scipy_tolerances, gradient=gradient, hessian=NOT_SUPPORTED,
                         minimizer_options=minimizer_options, tolerance=tolerance, verbosity=verbosity,
                         maxiter=maxiter,
                         strategy=strategy, criterion=criterion, name=name)


class ScipyNelderMeadV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 adaptive: Optional[bool] = True,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy Nelder-Mead V1"):
        options = {}
        minimizer_options = {}

        if adaptive is not None:
            options['adaptive'] = adaptive
        if options:
            minimizer_options['options'] = options

        scipy_tolerances = {'fatol': None, 'xatol': None}

        method = "Nelder-Mead"
        super().__init__(method=method, internal_tolerances=scipy_tolerances, gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED, minimizer_options=minimizer_options, tolerance=tolerance,
                         maxiter=maxiter,
                         verbosity=verbosity, strategy=strategy, criterion=criterion, name=name)


def combine_optimize_results(results):
    if len(results) == 1:
        return results[0]
    result = results[-1]
    for field in ['nfev', 'njev', 'nhev', 'nit']:
        if field in result:
            result[field] = sum((res[field] for res in results))
    return result

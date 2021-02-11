#  Copyright (c) 2021 zfit
import copy
import inspect
import math
import warnings
from typing import Optional, Dict, Callable, Union, Mapping

import numpy as np
import scipy.optimize  # pylint: disable=g-import-not-at-top
from scipy.optimize import SR1, LbfgsInvHessProduct

from .baseminimizer import BaseMinimizer, ZfitStrategy, minimize_supports
from .evaluation import LossEval
from .fitresult import FitResult
from .termination import ConvergenceCriterion, CRITERION_NOT_AVAILABLE, EDM
from ..core.parameter import set_values
from ..settings import run

NOT_SUPPORTED = object()


class ScipyBaseMinimizer(BaseMinimizer):
    def __init__(self,
                 method: str,
                 scipy_tolerances: Mapping[str, Optional[float]],
                 gradient: Optional[Union[Callable, str]],
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]],
                 minimizer_options: Mapping[str, object],
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="ScipyMinimizer"):
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
            minimizer_options['hesse'] = hessian
        self._scipy_tolerances = scipy_tolerances
        self._scipy_niter = 20
        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_options=minimizer_options,
                         strategy=strategy, criterion=criterion)

    @minimize_supports()
    def _minimize(self, loss, params, init):
        previous_result = init
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        minimizer_options = self.minimizer_options.copy()
        minimizer_options['bounds'] = limits

        use_gradient = 'grad' in minimizer_options
        if use_gradient:
            gradient = minimizer_options.pop('grad')
            gradient = evaluator.gradient if gradient == 'zfit' else gradient
            minimizer_options['jac'] = gradient

        use_hessian = 'hesse' in minimizer_options
        if use_hessian:
            hessian = minimizer_options.pop('hesse')
            hessian = evaluator.hessian if hessian == 'zfit' else hessian
            minimizer_options['hess'] = hessian

            is_update_strat = inspect.isclass(hessian) and issubclass(hessian,
                                                                      scipy.optimize.HessianUpdateStrategy)

            init_scale = 'auto'
            if previous_result:
                optimize_results = previous_result.info['original']
                inv_hessian = optimize_results.get('hess_inv')
                if inv_hessian is None:
                    hessian_init = optimize_results.get('hess')
                    if hessian_init is not None:
                        inv_hessian = np.linalg.inv(hessian_init)
                        init_scale = inv_hessian

        # tolerances and criterion
        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)

        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        scipy_tol = self._scipy_tolerances
        scipy_tol = {tol: init_tol if init is None else init for tol, init in scipy_tol.items()}

        valid = None
        optimize_results = None
        for i in range(self._scipy_niter):

            # update from previous run/result
            if use_hessian and is_update_strat:
                if not isinstance(init_scale, str):
                    init_scale = np.mean(init_scale)
                minimizer_options['hess'] = hessian(init_scale=init_scale)

            for tol, val in scipy_tol.items():
                minimizer_options['options'][tol] = val
            # perform minimization
            optim_result = scipy.optimize.minimize(fun=evaluator.value,
                                                   x0=init_values, **minimizer_options)
            xvalues = optim_result['x']
            if use_gradient:
                gradient = optim_result.get('jac')
            if use_hessian:
                inv_hessian = optim_result.get('hess_inv')
                if inv_hessian is None:
                    hessian_result = optim_result.get('hess')
                    if hessian_result is not None:
                        inv_hessian = np.linalg.inv(hessian_result)
                elif isinstance(inv_hessian, LbfgsInvHessProduct):
                    inv_hessian = inv_hessian.todense()

            fmin = optim_result.fun

            optimize_results = combine_optimize_results(
                [optim_result] if optimize_results is None else [optimize_results, optim_result])
            result_prelim = FitResult.from_scipy(loss=loss, params=params, result=optimize_results, minimizer=self,
                                                 edm=CRITERION_NOT_AVAILABLE, valid=valid)
            converged = criterion.converged(result_prelim)
            criterion_val = criterion.last_value

            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, {criterion.name}={criterion.last_value}"
                      f" {f', {tol}={val}'}" for tol, val in scipy_tol.items())

            if converged:
                break

            if use_hessian:
                if inv_hessian is not None:
                    init_scale = inv_hessian
            init_values = xvalues

            # update the tolerances
            tol_factor = min([max([self.tolerance / criterion_val * 0.3, 1e-2]), 0.2])
            for tol in scipy_tol:
                scipy_tol[tol] *= tol_factor

        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=optimize_results,
            minimizer=self,
            valid=valid,
            criterion=criterion,
            edm=criterion_val if type(criterion) == EDM else CRITERION_NOT_AVAILABLE,
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
        fitresult = FitResult.from_scipy(loss=loss, params=params, result=result, minimizer=self)
        # criterion.convergedV1(value=fitresult.fmin,
        #                            xvalues=xvalues,
        #                            grad=result['jac'],
        #                            inv_hesse=result['hess_inv'].todense())

        return fitresult


class ScipyLBFGSBV1(ScipyBaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: Optional[int] = None,
                 maxls: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
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

        scipy_tolerances = {'gtol': None, 'gtol': None}

        super().__init__(method="L-BFGS-B", tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=NOT_SUPPORTED,
                         criterion=criterion, scipy_tolerances=scipy_tolerances,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyTrustKrylovV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 inexact: Optional[bool] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
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

        super().__init__(method="trust-constr", tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=hessian,
                         criterion=criterion, scipy_tolerances=scipy_tolerances,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyTrustNCGV1(ScipyBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
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

        super().__init__(method="trust-constr", tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=hessian,
                         criterion=criterion, scipy_tolerances=scipy_tolerances,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyTrustConstrV1(ScipyBaseMinimizer):
    def __init__(self, tolerance: float = None,
                 initial_tr_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
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

        super().__init__(method="trust-constr", tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=hessian,
                         criterion=criterion, scipy_tolerances=scipy_tolerances,
                         minimizer_options=minimizer_options,
                         name=name)


class ScipyDoglegV1(ScipyBaseMinimizer):
    def __init__(self, tolerance: float = None,
                 initial_trust_radius: Optional[int] = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = 'zfit',
                 hessian: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = SR1,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: ZfitStrategy = None,
                 verbosity: Optional[int] = None,
                 name="Scipy trust-constr V1"):
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

        super().__init__(method="trust-constr", tolerance=tolerance, verbosity=verbosity,
                         strategy=strategy, gradient=gradient, hessian=hessian,
                         criterion=criterion, scipy_tolerances=scipy_tolerances,
                         minimizer_options=minimizer_options,
                         name=name)


def combine_optimize_results(results):
    if len(results) == 1:
        return results[0]
    result = results[-1]
    for field in ['nfev', 'njev', 'nhev', 'nit']:
        if field in result:
            result[field] = sum((res[field] for res in results))
    return result

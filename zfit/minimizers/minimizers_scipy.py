#  Copyright (c) 2021 zfit
import inspect
import math
import warnings
from typing import Optional, Dict, Callable, Union

import numpy as np
import scipy.optimize  # pylint: disable=g-import-not-at-top

from .baseminimizer import BaseMinimizer, ZfitStrategy, minimize_supports
from .evaluation import LossEval
from .fitresult import FitResult
from .termination import EDM
from ..core.parameter import set_values
from ..settings import run


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


class ScipyLBFGSBV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: Optional[int] = None,
                 maxls: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="Scipy L-BFGS-B V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            self.hesse = scipy.optimize.BFGS
        else:
            if isinstance(hesse, scipy.optimize.HessianUpdateStrategy) and not inspect.isclass(hesse):
                raise ValueError("If `hesse` is a HessianUpdateStrategy, it has to be a class that takes `init_scale`,"
                                 " not an instance. For further modification of other initial parameters, make a"
                                 " subclass of the update strategy.")
            self.hesse = hesse
        self.maxcor = maxcor
        self.maxls = 30 if maxls is None else maxls
        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports()
    def _minimize(self, loss, params):
        previous_result = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        minimize_kwargs = {
            'jac': evaluator.gradient if self.grad == 'zfit' else self.grad,
            'method': 'L-BFGS-B',
            'bounds': limits,
        }
        init_scale = None
        if previous_result:
            optimize_result = previous_result['info']['original']
            hess_inv = optimize_result.get('hess_inv')
            if hess_inv is not None:
                init_scale = np.diag(hess_inv.todense())
        if init_scale is None:
            init_scale = 'auto'
        maxiter = self.maxiter
        criterion = self._convergence_criterion_cls(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = loss.errordef * self.tolerance
        ftol = init_tol
        gtol = init_tol
        results = []
        n_iter = 20
        valid = None
        for i in range(n_iter):

            if inspect.isclass(self.hesse) and issubclass(self.hesse, scipy.optimize.HessianUpdateStrategy):
                hesse = self.hesse(init_scale=init_scale)
            else:
                hesse = self.hesse
            minimize_kwargs['hess'] = hesse

            options = {'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            if self.maxls is not None:
                options['maxls'] = self.maxls
            if self.maxcor is not None:
                options['maxcor'] = self.maxcor

            minimize_kwargs['options'] = options
            result = scipy.optimize.minimize(fun=evaluator.value,
                                             x0=init_values, **minimize_kwargs)
            results.append(result)

            xvalues = result['x']
            grad = result['jac']
            inv_hesse = result['hess_inv'].todense()
            fmin = result.fun
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=grad,
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm}, ftol={ftol}, gtol={gtol}")

            if edm < self.tolerance:
                break
            init_scale = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm * 0.3, 1e-2]), 0.2])
            ftol *= tol_factor  # lower tolerances
            gtol *= tol_factor
        else:
            valid = "Invalid, EDM not reached"
        result = combine_optimize_result(results)
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=result,
            minimizer=self,
            valid=valid,
            edm=edm,
        )


class ScipyTrustNCGV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 eta: Optional[float] = None,
                 max_trust_radius: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="Scipy trust-ncg V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            self.hesse = 'zfit'
        else:
            if isinstance(hesse, scipy.optimize.HessianUpdateStrategy) and not inspect.isclass(hesse):
                raise ValueError("If `hesse` is a HessianUpdateStrategy, it has to be a class that takes `init_scale`,"
                                 " not an instance. For further modification of other initial parameters, make a"
                                 " subclass of the update strategy.")
            self.hesse = hesse
        self.eta = eta
        self.max_trust_radius = max_trust_radius
        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports()
    def _minimize(self, loss, params):
        previous_result = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        hesse = evaluator.hessian if self.hesse == 'zfit' else self.hesse
        minimize_kwargs = {
            'jac': evaluator.gradient if self.grad == 'zfit' else self.grad,
            'hess': hesse,
            'method': 'trust-ncg',
            'bounds': limits,
        }
        if previous_result:
            optimize_result = previous_result['info']['original']
            init_scale = np.diag(optimize_result['hess_inv'].todense())

        else:
            init_scale = 'auto'
        init_trust_radius = None
        criterion = self._convergence_criterion_cls(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        gtol = init_tol
        results = []
        n_iter = 20
        valid = None
        for i in range(n_iter):

            if inspect.isclass(self.hesse) and issubclass(self.hesse, scipy.optimize.HessianUpdateStrategy):
                hesse = self.hesse(init_scale=init_scale)

            minimize_kwargs['hess'] = hesse

            options = {'init_trust_radius': init_trust_radius, 'gtol': gtol}
            if self.max_trust_radius is not None:
                options['max_trust_radius'] = self.max_trust_radius
            if self.eta is not None:
                options['eta'] = self.eta

            minimize_kwargs['options'] = options
            result = scipy.optimize.minimize(fun=evaluator.value,
                                             x0=init_values, **minimize_kwargs)
            results.append(result)

            xvalues = result['x']
            grad = result['jac']
            inv_hesse = np.linalg.inv(result['hess'])
            fmin = result.fun
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=grad,
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  gtol={gtol}")

            if edm < self.tolerance:
                break
            init_scale = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.2])
            gtol *= tol_factor
        else:
            valid = "Invalid, EDM not reached"
        result = combine_optimize_result(results)
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=result,
            minimizer=self,
            valid=valid,
            edm=edm,
        )


class ScipyTrustKrylovV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 inexact: bool = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str, scipy.optimize.HessianUpdateStrategy]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="Scipy trust-krylov V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            self.hesse = 'zfit'
        else:
            if isinstance(hesse, scipy.optimize.HessianUpdateStrategy) and not inspect.isclass(hesse):
                raise ValueError("If `hesse` is a HessianUpdateStrategy, it has to be a class that takes `init_scale`,"
                                 " not an instance. For further modification of other initial parameters, make a"
                                 " subclass of the update strategy.")
            self.hesse = hesse
        self.inexact = inexact

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports()
    def _minimize(self, loss, params):
        previous_result = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        limits = [(run(p.lower), run(p.upper)) for p in params]
        init_values = np.array(run(params))

        minimize_kwargs = {
            'jac': evaluator.gradient if self.grad == 'zfit' else self.grad,
            'hess': evaluator.hessian if self.hesse == 'zfit' else self.hesse,
            'method': 'trust-krylov',
            'bounds': limits,
        }
        if previous_result:
            optimize_result = previous_result['info']['original']
            init_scale = np.diag(optimize_result['hess_inv'].todense())

        else:
            init_scale = 'auto'
        criterion = self._convergence_criterion_cls(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        gtol = init_tol
        results = []
        n_iter = 20
        valid = None
        for i in range(n_iter):

            # update if it is an instance of HessianUpdateStrategy
            if inspect.isclass(self.hesse) and issubclass(self.hesse, scipy.optimize.HessianUpdateStrategy):
                hesse = self.hesse(init_scale=init_scale)
                minimize_kwargs['hess'] = hesse

            options = {'inexact': self.inexact, 'gtol': gtol}

            minimize_kwargs['options'] = options
            result = scipy.optimize.minimize(fun=evaluator.value,
                                             x0=init_values, **minimize_kwargs)
            results.append(result)

            xvalues = result['x']
            grad = result['jac']
            hesse = result['hess']
            inv_hesse = np.linalg.inv(hesse)
            fmin = result.fun
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=grad,
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  gtol={gtol}")

            if edm < self.tolerance:
                break

            init_scale = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.2])
            gtol *= tol_factor
        else:
            valid = "Invalid, EDM not reached"
        result = combine_optimize_result(results)
        return FitResult.from_scipy(
            loss=loss,
            params=params,
            result=result,
            minimizer=self,
            valid=valid,
            edm=edm,
        )


def combine_optimize_result(results):
    result = results[-1]
    for field in ['nfev', 'njev', 'nhev', 'nit']:
        if field in result:
            result[field] = sum((res[field] for res in results))
    return result

#  Copyright (c) 2021 zfit
import math
from typing import Optional, Dict, Union, Callable

import nlopt
import numpy as np

from .baseminimizer import BaseMinimizer, ZfitStrategy, minimize_supports
from .evaluation import LossEval
from .fitresult import FitResult
from .termination import EDM, CRITERION_NOT_AVAILABLE
from ..core.parameter import set_values
from ..settings import run


class NLopt(BaseMinimizer):
    def __init__(self, algorithm: int = nlopt.LD_LBFGS, tolerance: Optional[float] = None,
                 strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5, name: Optional[str] = None,
                 maxiter: Optional[int] = None, minimizer_options: Optional[Dict[str, object]] = None):
        """NLopt contains multiple different optimization algorithms.py

        `NLopt <https://nlopt.readthedocs.io/en/latest/>`_ is a free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as original implementations of various other
        algorithms.

        Args:
            algorithm: Define which algorithm to be used. These are taken from `nlopt.ALGORITHM` (where `ALGORITHM` is
                the actual algorithm). A comprehensive list and description of all implemented algorithms is
                available `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.
                The wrapper is optimized for Local gradient-based optimization and may breaks with
                others. However, please post a feature request in case other algorithms are requested.

                The naming of the algorithm starts with either L/G (Local/Global)
                 and N/D (derivative-free/derivative-based).

                Local optimizer ptions include (but not only)

                Derivative free:
                - LN_NELDERMEAD: The Nelder Mead Simplex algorithm, which seems to perform not so well, but can be used
                  to narrow down a minimum.
                - LN_SBPLX: SubPlex is an improved version of the Simplex algorithms and usually performs better.

                With derivative:
                - LD_MMA: Method of Moving Asymptotes, an improved CCSA
                  ("conservative convex separable approximation") variant of the original MMA algorithm
                - LD_SLSQP: this is a sequential quadratic programming (SQP) algorithm for (nonlinearly constrained)
                  gradient-based optimization
                - LD_LBFGS: version of the famous low-storage BFGS algorithm, an approximate Newton method. The same as
                  the Minuit algorithm is built on.
                - LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, LD_TNEWTON_RESTART, LD_TNEWTON: a preconditioned
                  inexact truncated Newton algorithm. Multiple variations, with and without preconditioning and/or
                  restart are provided.
                - LD_VAR1, LD_VAR2: a shifted limited-memory variable-metric algorithm, either using a rank 1 or rank 2
                  method.

            tolerance (Union[float, None]):
            strategy (Union[None, None]):
            verbosity (int):
            name (Union[None, None]):
            maxiter (Union[None, None]):
            minimizer_options (Union[None, None]):
        """
        self.algorithm = algorithm
        super().__init__(name=name, tolerance=tolerance, verbosity=verbosity, minimizer_options=minimizer_options,
                         strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=False)
    def _minimize(self, loss, params):

        minimizer = nlopt.opt(self.algorithm, len(params))

        n_eval = 0

        init_val = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))
        minimizer.set_ftol_abs(self.tolerance)

        for name, value in self.minimizer_options:
            minimizer.set_param(name, value)

        xvalues = minimizer.optimize(init_val)
        set_values(params, xvalues)
        edm = -999
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, params=params, xvalues=xvalues)


class NLoptLBFGSV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt L-BFGS V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse
        self.maxcor = maxcor

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss

            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)
        minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))
        if self.maxcor is not None:
            minimizer.set_vector_storage(self.maxcor)

        for name, value in self.minimizer_options:
            minimizer.set_param(name, value)

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            result = FitResult.from_nlopt(loss, minimizer=self, opt=minimizer,
                                 edm=CRITERION_NOT_AVAILABLE, n_eval=n_iter,
                                          params=params,
                                 xvalues=xvalues, inv_hess=inv_hesse,
                                 valid=valid)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if criterion.converged(result):
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"

        if type(criterion) == EDM:
            edm = criterion.last_value
        else:
            edm = CRITERION_NOT_AVAILABLE
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid, criterion=criterion)


class NLoptTruncNewtonV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt Precond trunc Newton V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse
        self.maxcor = maxcor

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)
        minimizer = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))
        if self.maxcor is not None:
            minimizer.set_vector_storage(self.maxcor)

        for name, value in self.minimizer_options:
            minimizer.set_param(name, value)

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptSLSQPV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt SLSQP V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)

        minimizer = nlopt.opt(nlopt.LD_SLSQP, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptMMAV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt MMA V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)

        minimizer = nlopt.opt(nlopt.LD_MMA, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptCCSAQV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt MMA V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)

        minimizer = nlopt.opt(nlopt.LD_CCSAQ, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)

class NLoptSubplexV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt Subplex V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)

        minimizer = nlopt.opt(nlopt.LN_SBPLX, len(params))

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)
            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


# NOT WORKING!
class NLoptMLSLV1(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[int] = None,
                 name="NLopt MLSL V1"):
        self.grad = 'zfit' if grad is None else grad
        if hesse is None:
            hesse = 'zfit'

        self.hesse = hesse

        super().__init__(name, tolerance, verbosity, minimizer_options={}, strategy=strategy, maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params):
        inv_hesse = None
        if isinstance(loss, FitResult):
            previous_result = loss
            loss = previous_result.loss
            inv_hesse = previous_result.info.get('inv_hesse')

        if inv_hesse is None:
            step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])
        else:
            step_sizes = np.diag(inv_hesse)

        minimizer = nlopt.opt(nlopt.GD_MLSL_LDS, len(params))
        local_minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))
        minimizer.set_local_optimizer(local_minimizer)

        init_values = np.array(run(params))

        evaluator = LossEval(loss=loss,
                             params=params,
                             strategy=self.strategy,
                             do_print=self.verbosity > 8,
                             minimizer=self)

        func = evaluator.value
        grad_value_func = evaluator.value_gradients

        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = grad_value_func(x)
                grad[:] = np.array(run(gradients))
            else:
                value = func(x)

            return value

        minimizer.set_min_objective(obj_func)

        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        init_tol *= 10
        ftol = init_tol
        xtol = init_tol ** 0.5
        n_iter_max = 20
        valid = None
        n_iter = 0
        edm = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if edm is not None:
                tol_factor_full = self.tolerance / edm
                if tol_factor_full < 1e-8:
                    minimizer.set_ftol_rel(ftol)
                    minimizer.set_xtol_rel(xtol)

            minimizer.set_initial_step(step_sizes)

            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            n_iter += minimizer.get_numevals()
            hesse = evaluator.hessian(params)
            inv_hesse = np.linalg.inv(hesse)
            edm = criterion.calculateV1(value=fmin, xvalues=xvalues, grad=lambda: evaluator.gradient(params),
                                        inv_hesse=inv_hesse)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if edm < self.tolerance:
                break

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / edm, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, n_eval=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)

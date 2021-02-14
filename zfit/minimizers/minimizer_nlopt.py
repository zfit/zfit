#  Copyright (c) 2021 zfit
import copy
import math
from typing import Optional, Dict, Union, Callable, Mapping

import nlopt
import numpy as np

from .baseminimizer import BaseMinimizer, minimize_supports, NOT_SUPPORTED, print_minimization_status
from .evaluation import LossEval
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import EDM, CRITERION_NOT_AVAILABLE, ConvergenceCriterion
from ..core.parameter import set_values
from ..settings import run


class NLopt(BaseMinimizer):
    def __init__(self, algorithm: int = nlopt.LD_LBFGS, tolerance: Optional[float] = None,
                 strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5, name: Optional[str] = None,
                 maxiter: Optional[Union[int, str]] = 'auto', minimizer_options: Optional[Dict[str, object]] = None):
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


class NLoptBaseMinimizer(BaseMinimizer):
    _ALL_NLOPT_TOL = 'fatol', 'ftol', 'xatol', 'xtol'

    def __init__(self,
                 algorithm: int,
                 gradient: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 hessian: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 maxiter: Optional[Union[int, str]],
                 minimizer_options: Mapping[str, object],
                 internal_tolerances: Mapping[str, Optional[float]] = None,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name: str = "NLopt Base Minimizer V1"):

        self._algorithm = algorithm

        minimizer_options = copy.copy(minimizer_options)

        if gradient is not NOT_SUPPORTED:
            if gradient is False:
                raise ValueError("grad cannot be False for NLopt minimizer.")
            minimizer_options['gradient'] = gradient
        if hessian is not NOT_SUPPORTED:
            minimizer_options['hessian'] = hessian

        if internal_tolerances is None:
            internal_tolerances = {}
        else:
            internal_tolerances = copy.copy(internal_tolerances)
        for tol in self._ALL_NLOPT_TOL:
            if tol not in internal_tolerances:
                internal_tolerances[tol] = None
        self._internal_tolerances = internal_tolerances
        self._internal_maxiter = 20

        super().__init__(name=name,
                         tolerance=tolerance,
                         verbosity=verbosity,
                         minimizer_options=minimizer_options,
                         strategy=strategy,
                         criterion=criterion,
                         maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params, init):
        previous_result = init
        evaluator = self.create_evaluator(loss, params)

        # create minimizer instance
        minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))

        # initial values as array
        init_values = np.array(run(params))

        # get and set the limits
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)

        # create and set objective function. Either returns only the value or the value
        # and sets the gradient in-place
        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = evaluator.value_gradients(x)
                grad[:] = np.array(run(gradients))
            else:
                value = evaluator.value(x)

            return value

        minimizer.set_min_objective(obj_func)

        # set maximum number of iterations, also set in evaluator
        minimizer.set_maxeval(self.get_maxiter(len(params)))

        inv_hesse = None
        hessian_init = None
        init_scale = None
        if previous_result:
            if previous_result:

                inv_hessian = previous_result.info.get('inv_hesse')
                if inv_hessian is None:
                    hessian_init = previous_result.info.get('hesse')
                    if hessian_init is not None:
                        inv_hessian = np.linalg.inv(hessian_init)
                        init_scale = np.diag(inv_hessian)

        minimizer_options = self.minimizer_options.copy()
        local_minimizer = None
        local_minimizer_options = minimizer_options.pop("local_minimizer_options", None)
        if local_minimizer_options is not None:
            local_minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))
            local_minimizer.set_lower_bounds(lower)
            local_minimizer.set_upper_bounds(upper)

        maxcor = minimizer_options.pop('maxcor', None)
        if maxcor is not None:
            minimizer.set_vector_storage(maxcor)

        for name, value in minimizer_options.items():
            minimizer.set_param(name, value)

        criterion = self.criterion(tolerance=self.tolerance, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tolerance), loss.errordef * self.tolerance * 1e3])
        # init_tol *= 10
        internal_tol = self._internal_tolerances
        internal_tol = {tol: init_tol if init is None else init for tol, init in internal_tol.items()}
        if 'xtol' in internal_tol:
            internal_tol['xtol'] **= 0.5

        valid = None
        edm = None
        criterion_value = None
        for i in range(self._internal_maxiter):

            self._set_tolerances_inplace(minimizer=minimizer,
                                         internal_tol=internal_tol,
                                         criterion_value=criterion_value)

            if local_minimizer is not None:
                self._set_tolerances_inplace(minimizer=local_minimizer,
                                             internal_tol=internal_tol,
                                             criterion_value=criterion_value)

                minimizer.set_local_optimizer(local_minimizer)

            if init_scale is None:
                step_sizes = np.array([p.step_size if p.has_step_size else 0 for p in params])  # TODO: is 0 okay?
            else:
                step_sizes = init_scale
            minimizer.set_initial_step(step_sizes)

            # run the minimization
            xvalues = minimizer.optimize(init_values)
            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()

            result_prelim = FitResult.from_nlopt(loss,
                                                 minimizer=self, opt=minimizer,
                                                 edm=CRITERION_NOT_AVAILABLE, niter=evaluator.nfunc_eval,
                                                 params=params,
                                                 xvalues=xvalues,
                                                 valid=valid)
            converged = criterion.converged(result_prelim)
            criterion_value = criterion.last_value
            if isinstance(criterion, EDM):
                edm = criterion.last_value
            else:
                edm = CRITERION_NOT_AVAILABLE

            if self.verbosity > 5:
                print_minimization_status(converged=converged, criterion=criterion, evaluator=evaluator, i=i, fmin=fmin,
                                          internal_tol=internal_tol)
            if (evaluator.current_hesse_value is not None
                and (hessian_init is None
                     or not np.allclose(hessian_init,
                                        evaluator.current_hesse_value))):  # either non-existent or different
                hessian_init = evaluator.current_hesse_value
            else:
                hessian_init = None  # the hessian we have is not valid anymore

            if converged:
                break

            if hessian_init is not None:
                inv_hesse = np.linalg.inv(hessian_init)
                init_scale = np.diag(inv_hesse)
            init_values = xvalues

            # update the tolerances
            self._update_tol_inplace(criterion_value=criterion_value, internal_tol=internal_tol)

        else:
            valid = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tolerance} not reached."

        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm,
                                    niter=evaluator.niter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid, criterion=criterion)

    def _set_tolerances_inplace(self, minimizer, internal_tol, criterion_value):
        # set all the tolerances
        fatol = internal_tol.get('fatol')
        if fatol is not None:
            minimizer.set_ftol_abs(fatol)
        xatol = internal_tol.get('xatol')
        if xatol is not None:
            # minimizer.set_xtol_abs([xatol] * len(params))
            minimizer.set_xtol_abs(xatol)
        # set relative tolerances later as it can be unstable. Just use them when approaching
        if criterion_value is not None:
            tol_factor_full = self.tolerance / criterion_value
            if tol_factor_full < 1e-8:
                ftol = internal_tol.get('ftol')
                if ftol is not None:
                    minimizer.set_ftol_rel(ftol)

                xtol = internal_tol.get('xtol')
                if xtol is not None:
                    # minimizer.set_xtol_rel([xtol] * len(params))  # TODO: one value or vector?
                    minimizer.set_xtol_rel(xtol)  # TODO: one value or vector?


class NLoptLBFGSV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt L-BFGS V1"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_LBFGS,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={'maxcor': maxcor},
                         strategy=strategy,
                         maxiter=maxiter)

        @property
        def maxcor(self):
            return self.minimizer_options.get('maxcor')


class NLoptLBFGSV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        criterion_value = None
        for i in range(n_iter_max):

            minimizer.set_ftol_abs(ftol)
            # minimizer.set_xtol_abs(xtol * 10)
            if criterion_value is not None:
                tol_factor_full = self.tolerance / criterion_value
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
                                          edm=CRITERION_NOT_AVAILABLE, niter=n_iter,
                                          params=params,
                                          xvalues=xvalues, inv_hess=inv_hesse,
                                          valid=valid)
            if self.verbosity > 5:
                print(f"Finished iteration {i}, fmin={fmin}, edm={edm},  ftol={ftol}, xtol={xtol}")

            if criterion.converged(result):
                break

            if type(criterion) == EDM:
                edm = criterion.last_value
                criterion_value = edm
            else:
                edm = CRITERION_NOT_AVAILABLE
                criterion_value = criterion.last_value

            step_sizes = np.diag(inv_hesse)
            init_values = xvalues
            tol_factor = min([max([self.tolerance / criterion_value, 1e-2]), 0.35])
            ftol *= tol_factor
            xtol *= tol_factor ** 0.5
        else:
            valid = "Invalid, EDM not reached"

        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid, criterion=criterion)


class NLoptTruncNewtonV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt Truncated Newton"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_TNEWTON_PRECOND_RESTART,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={'maxcor': maxcor},
                         strategy=strategy,
                         maxiter=maxiter)

        @property
        def maxcor(self):
            return self.minimizer_options.get('maxcor')


class NLoptTruncNewtonV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptSLSQPV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt SLSQP"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_SLSQP,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptSLSQPV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptMMAV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt MMA"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_MMA,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptMMAV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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

            # minimizer.set_initial_step(step_sizes, init_values)

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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptCCSAQV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt CCSAQ"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_CCSAQ,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptCCSAQV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptSubplexV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt Subplex"):
        super().__init__(name=name,
                         algorithm=nlopt.LN_SBPLX,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptSubplexV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)


class NLoptMLSLV1(NLoptBaseMinimizer):
    def __init__(self,
                 tolerance: float = None,
                 local_minimizer: Optional[Union[int, Mapping[str, object]]] = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt MLSL"):
        local_minimizer = nlopt.LD_LBFGS if local_minimizer is None else local_minimizer
        if not isinstance(local_minimizer, Mapping):
            local_minimizer = {'algorithm': local_minimizer}
        if 'algorithm' not in local_minimizer:
            raise ValueError("algorithm needs to be specified in local_minimizer")

        minimizer_options = {'local_minimizer_options': local_minimizer}
        super().__init__(name=name,
                         algorithm=nlopt.GD_MLSL_LDS,
                         tolerance=tolerance,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options=minimizer_options,
                         strategy=strategy,
                         maxiter=maxiter)


# NOT WORKING!
class NLoptMLSLV1old(BaseMinimizer):

    def __init__(self,
                 tolerance: float = None,
                 verbosity: Optional[int] = None,
                 grad: Optional[Union[Callable, str]] = None,
                 hesse: Optional[Union[Callable, str]] = None,
                 strategy: ZfitStrategy = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
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
        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, niter=n_iter, params=params,
                                    xvalues=xvalues, inv_hess=inv_hesse, valid=valid)

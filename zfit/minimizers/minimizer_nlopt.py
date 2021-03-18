#  Copyright (c) 2021 zfit
import collections
import copy
import math
from typing import Callable, Dict, Mapping, Optional, Union

import nlopt
import numpy as np

from ..core.parameter import set_values
from ..settings import run
from ..util.exception import MaximumIterationReached
from .baseminimizer import (NOT_SUPPORTED, BaseMinimizer, minimize_supports,
                            print_minimization_status)
from .evaluation import LossEval
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import CRITERION_NOT_AVAILABLE, EDM, ConvergenceCriterion

# class NLopt(BaseMinimizer):
#     def __init__(self, algorithm: int = nlopt.LD_LBFGS, tol: Optional[float] = None,
#                  strategy: Optional[ZfitStrategy] = None, verbosity: Optional[int] = 5, name: Optional[str] = None,
#                  maxiter: Optional[Union[int, str]] = 'auto', minimizer_options: Optional[Dict[str, object]] = None):
#         """NLopt contains multiple different optimization algorithms.py
#
#         `NLopt <https://nlopt.readthedocs.io/en/latest/>`_ is a free/open-source library for nonlinear optimization,
#         providing a common interface for a number of
#         different free optimization routines available online as well as original implementations of various other
#         algorithms.
#
#         Args:
#             algorithm: Define which algorithm to be used. These are taken from `nlopt.ALGORITHM` (where `ALGORITHM` is
#                 the actual algorithm). A comprehensive list and description of all implemented algorithms is
#                 available `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.
#                 The wrapper is optimized for Local gradient-based optimization and may breaks with
#                 others. However, please post a feature request in case other algorithms are requested.
#
#                 The naming of the algorithm starts with either L/G (Local/Global)
#                  and N/D (derivative-free/derivative-based).
#
#                 Local optimizer ptions include (but not only)
#
#                 Derivative free:
#                 - LN_NELDERMEAD: The Nelder Mead Simplex algorithm, which seems to perform not so well, but can be used
#                   to narrow down a minimum.
#                 - LN_SBPLX: SubPlex is an improved version of the Simplex algorithms and usually performs better.
#
#                 With derivative:
#                 - LD_MMA: Method of Moving Asymptotes, an improved CCSA
#                   ("conservative convex separable approximation") variant of the original MMA algorithm
#                 - LD_SLSQP: this is a sequential quadratic programming (SQP) algorithm for (nonlinearly constrained)
#                   gradient-based optimization
#                 - LD_LBFGS: version of the famous low-storage BFGS algorithm, an approximate Newton method. The same as
#                   the Minuit algorithm is built on.
#                 - LD_TNEWTON_PRECOND_RESTART, LD_TNEWTON_PRECOND, LD_TNEWTON_RESTART, LD_TNEWTON: a preconditioned
#                   inexact truncated Newton algorithm. Multiple variations, with and without preconditioning and/or
#                   restart are provided.
#                 - LD_VAR1, LD_VAR2: a shifted limited-memory variable-metric algorithm, either using a rank 1 or rank 2
#                   method.
#
#             tol (Union[float, None]):
#             strategy (Union[None, None]):
#             verbosity (int):
#             name (Union[None, None]):
#             maxiter (Union[None, None]):
#             minimizer_options (Union[None, None]):
#         """
#         self.algorithm = algorithm
#         super().__init__(name=name, tol=tol, verbosity=verbosity, minimizer_options=minimizer_options,
#                          strategy=strategy, maxiter=maxiter)
#
#     @minimize_supports(init=False)
#     def _minimize(self, loss, params):
#
#         minimizer = nlopt.opt(self.algorithm, len(params))
#
#         n_eval = 0
#
#         init_val = np.array(run(params))
#
#         evaluator = LossEval(loss=loss,
#                              params=params,
#                              strategy=self.strategy,
#                              do_print=self.verbosity > 8)
#
#         func = evaluator.value
#         grad_value_func = evaluator.value_gradients
#
#         def obj_func(x, grad):
#             if grad.size > 0:
#                 value, gradients = grad_value_func(x)
#                 grad[:] = np.array(run(gradients))
#             else:
#                 value = func(x)
#
#             return value
#
#         minimizer.set_min_objective(obj_func)
#         lower = np.array([p.lower for p in params])
#         upper = np.array([p.upper for p in params])
#         minimizer.set_lower_bounds(lower)
#         minimizer.set_upper_bounds(upper)
#         minimizer.set_maxeval(self.get_maxiter(len(params)))
#         minimizer.set_ftol_abs(self.tol)
#
#         for name, value in self.minimizer_options:
#             minimizer.set_param(name, value)
#
#         xvalues = minimizer.optimize(init_val)
#         set_values(params, xvalues)
#         edm = -999
#         return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm, params=params, xvalues=xvalues)


class NLoptBaseMinimizerV1(BaseMinimizer):
    _ALL_NLOPT_TOL = (
        # 'fatol',
        'ftol',
        'xatol',
        'xtol'
    )

    def __init__(self,
                 algorithm: int,
                 gradient: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 hessian: Optional[Union[Callable, str, NOT_SUPPORTED]],
                 maxiter: Optional[Union[int, str]],
                 minimizer_options: Optional[Mapping[str, object]],
                 internal_tols: Mapping[str, Optional[float]] = None,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name: str = "NLopt Base Minimizer V1"):
        """NLopt contains multiple different optimization algorithms.py.

        `NLopt <https://nlopt.readthedocs.io/en/latest/>`_ is a free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as original implementations of various other
        algorithms.

        Args:
            algorithm: Define which algorithm to be used. These are taken from `nlopt.ALGORITHM` (where `ALGORITHM` is
                the actual algorithm). A comprehensive list and description of all implemented algorithms is
                available `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.
                The wrapper is optimized for Local gradient-based optimization and may break with
                others. However, please post a feature request in case other algorithms are requested.

                The naming of the algorithm starts with either L/G (Local/Global)
                 and N/D (derivative-free/derivative-based).

                Local optimizer options include (but are not limited to)

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

            tol (Union[float, None]):
            strategy (Union[None, None]):
            verbosity (int):
            name (Union[None, None]):
            maxiter (Union[None, None]):
            minimizer_options (Union[None, None]):
        """
        self._algorithm = algorithm

        minimizer_options = copy.copy(minimizer_options)

        if gradient is not NOT_SUPPORTED:
            if gradient is False:
                raise ValueError("grad cannot be False for NLopt minimizer.")
            minimizer_options['gradient'] = gradient
        if hessian is not NOT_SUPPORTED:
            minimizer_options['hessian'] = hessian

        if internal_tols is None:
            internal_tols = {}
        else:
            internal_tols = copy.copy(internal_tols)
        for nlopt_tol in self._ALL_NLOPT_TOL:
            if nlopt_tol not in internal_tols:
                internal_tols[nlopt_tol] = None
        self._internal_tols = internal_tols

        # private kept variables
        self._internal_maxiter = 20
        self._nrandom_max = 5

        super().__init__(name=name,
                         tol=tol,
                         verbosity=verbosity,
                         minimizer_options=minimizer_options,
                         strategy=strategy,
                         criterion=criterion,
                         maxiter=maxiter)

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init):
        previous_result = init
        if init:
            set_values(params=params, values=init)
        evaluator = self.create_evaluator(loss, params)

        # create minimizer instance
        minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))

        # initial values as array
        xvalues = np.array(run(params))

        # get and set the limits
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        minimizer.set_lower_bounds(lower)
        minimizer.set_upper_bounds(upper)

        # create and set objective function. Either returns only the value or the value
        # and sets the gradient in-place
        def obj_func(x, grad):
            if grad.size > 0:
                value, gradients = evaluator.value_gradient(x)
                grad[:] = np.array(run(gradients))
            else:
                value = evaluator.value(x)

            return value

        minimizer.set_min_objective(obj_func)

        # set maximum number of iterations, also set in evaluator
        minimizer.set_maxeval(self.get_maxiter(len(params)))

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

        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        init_tol = min([math.sqrt(loss.errordef * self.tol), loss.errordef * self.tol * 1e3])
        # init_tol *= 10
        internal_tol = self._internal_tols
        internal_tol = {tol: init_tol if init is None else init for tol, init in internal_tol.items()}
        if 'xtol' in internal_tol:
            internal_tol['xtol'] **= 0.5
        if 'ftol' in internal_tol:
            internal_tol['ftol'] **= 0.5

        valid = None
        edm = None
        criterion_value = None
        maxiter_reached = False
        valid_message = ""
        result_prelim = previous_result
        nrandom = 0
        for i in range(self._internal_maxiter):
            init_scale = []
            approx_step_sizes = {}
            if result_prelim:
                approx_step_sizes = result_prelim.hesse(params=params, method='approx')
            empty_dict = {}
            for param in params:
                step_size = approx_step_sizes.get(param, empty_dict).get('error')
                if step_size is None and param.has_step_size:
                    step_size = param.step_size
                init_scale.append(step_size)

            minimizer.set_initial_step(init_scale)

            self._set_tols_inplace(minimizer=minimizer,
                                   internal_tol=internal_tol,
                                   criterion_value=criterion_value)

            # some (global) optimizers use a local minimizer, set that here
            if local_minimizer is not None:
                self._set_tols_inplace(minimizer=local_minimizer,
                                       internal_tol=internal_tol,
                                       criterion_value=criterion_value)

                minimizer.set_local_optimizer(local_minimizer)

            # run the minimization
            try:
                xvalues = minimizer.optimize(xvalues)
            except MaximumIterationReached:
                maxiter_reached = True
                valid = False
                valid_message = "Maxiter reached, terminated without convergence"
            except RuntimeError:
                if self.verbosity > 5:
                    print("Minimization in NLopt failed, restarting with slightly varied parameters.")
                if nrandom < self._nrandom_max:  # in order not to start too close
                    init_scale_no_nan = np.nan_to_num(init_scale, nan=1.)
                    xvalues += np.random.uniform(low=-init_scale_no_nan, high=init_scale_no_nan) / 2
                    nrandom += 1
            else:
                maxiter_reached = evaluator.niter > evaluator.maxiter

            set_values(params, xvalues)
            fmin = minimizer.last_optimum_value()  # TODO: what happens if minimization terminated?
            with evaluator.ignore_maxiter():
                result_prelim = FitResult.from_nlopt(loss,
                                                     minimizer=self,
                                                     opt=minimizer,
                                                     edm=CRITERION_NOT_AVAILABLE, niter=evaluator.nfunc_eval,
                                                     params=params,
                                                     evaluator=evaluator,
                                                     criterion=None,
                                                     xvalues=xvalues,
                                                     valid=valid, message=valid_message)
                converged = criterion.converged(result_prelim)
                valid = converged
            criterion_value = criterion.last_value
            if isinstance(criterion, EDM):
                edm = criterion.last_value
            else:
                edm = CRITERION_NOT_AVAILABLE

            if self.verbosity > 5:
                print_minimization_status(converged=converged, criterion=criterion, evaluator=evaluator, i=i, fmin=fmin,
                                          internal_tol=internal_tol)

            if converged or maxiter_reached:
                break

            # update the tols
            self._update_tol_inplace(criterion_value=criterion_value, internal_tol=internal_tol)

        else:
            valid = False
            valid_message = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tol} not reached."

        return FitResult.from_nlopt(loss, minimizer=self.copy(), opt=minimizer, edm=edm,
                                    niter=evaluator.niter, params=params, evaluator=evaluator,
                                    xvalues=xvalues, valid=valid, criterion=criterion,
                                    message=valid_message)

    def _set_tols_inplace(self, minimizer, internal_tol, criterion_value):
        # set all the tolerances
        fatol = internal_tol.get('fatol')
        if fatol is not None:
            minimizer.set_ftol_abs(fatol ** 0.5)
        xatol = internal_tol.get('xatol')
        if xatol is not None:
            # minimizer.set_xtol_abs([xatol] * len(params))
            minimizer.set_xtol_abs(xatol)
        # set relative tolerances later as it can be unstable. Just use them when approaching
        if criterion_value is not None:
            tol_factor_full = self.tol / criterion_value
            if tol_factor_full < 1e-8:
                ftol = internal_tol.get('ftol')
                if ftol is not None:
                    minimizer.set_ftol_rel(ftol)

                xtol = internal_tol.get('xtol')
                if xtol is not None:
                    # minimizer.set_xtol_rel([xtol] * len(params))  # TODO: one value or vector?
                    minimizer.set_xtol_rel(xtol)  # TODO: one value or vector?


class NLoptLBFGSV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt L-BFGS V1"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_LBFGS,
                         tol=tol,
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


class NLoptTruncNewtonV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 maxcor: bool = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt Truncated Newton"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_TNEWTON_PRECOND_RESTART,
                         tol=tol,
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


class NLoptSLSQPV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt SLSQP"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_SLSQP,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptMMAV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt MMA"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_MMA,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptCCSAQV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt CCSAQ"):
        super().__init__(name=name,
                         algorithm=nlopt.LD_CCSAQ,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptSubplexV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt Subplex"):
        super().__init__(name=name,
                         algorithm=nlopt.LN_SBPLX,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options={},
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptMLSLV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 randomized: bool = None,
                 local_minimizer: Optional[Union[int, Mapping[str, object]]] = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt MLSL",
                 ):
        if randomized is None:
            randomized = False
        if randomized:
            algorithm = nlopt.GD_MLSL_LDS
        else:
            algorithm = nlopt.GD_MLSL

        local_minimizer = nlopt.LD_LBFGS if local_minimizer is None else local_minimizer
        if not isinstance(local_minimizer, collections.Mapping):
            local_minimizer = {'algorithm': local_minimizer}
        if 'algorithm' not in local_minimizer:
            raise ValueError("algorithm needs to be specified in local_minimizer")

        minimizer_options = {'local_minimizer_options': local_minimizer}
        super().__init__(name=name,
                         algorithm=algorithm,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options=minimizer_options,
                         strategy=strategy,
                         maxiter=maxiter)


class NLoptStoGOV1(NLoptBaseMinimizerV1):
    def __init__(self,
                 tol: float = None,
                 randomized: bool = None,
                 verbosity: Optional[int] = None,
                 maxiter: Optional[Union[int, str]] = 'auto',
                 strategy: ZfitStrategy = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 name="NLopt MLSL"):

        if randomized is None:
            randomized = False

        if randomized:
            algorithm = nlopt.GD_STOGO_RAND
        else:
            algorithm = nlopt.GD_STOGO
        super().__init__(name=name,
                         algorithm=algorithm,
                         tol=tol,
                         gradient=NOT_SUPPORTED,
                         hessian=NOT_SUPPORTED,
                         criterion=criterion,
                         verbosity=verbosity,
                         minimizer_options=None,
                         strategy=strategy,
                         maxiter=maxiter)

#  Copyright (c) 2021 zfit
import math
from typing import Optional, Union, Callable, Dict

import ipyopt
import numpy as np

from .baseminimizer import BaseMinimizer, minimize_supports, print_minimization_status
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import ConvergenceCriterion, EDM, CRITERION_NOT_AVAILABLE
from ..core.parameter import set_values
# class IPopt(ScipyBaseMinimizer):
#
#     def __init__(self,
#                  tol: float = None,
#                  # maxcor: Optional[int] = None,
#                  # maxls: Optional[int] = None,
#                  verbosity: Optional[int] = None,
#                  gradient: Optional[Union[Callable, str]] = 'zfit',
#                  maxiter: Optional[Union[int, str]] = 'auto',
#                  criterion: Optional[ConvergenceCriterion] = None,
#                  strategy: Optional[ZfitStrategy] = None,
#                  name="IPopt"):
#         options = {'linear_solver': "mumps"}
#
#         minimizer_options = {}
#         if options:
#             minimizer_options['options'] = options
#
#         scipy_tols = {'tol': None}
#
#         super().__init__(method=None, internal_tols=scipy_tols, gradient=gradient,
#                          hessian=NOT_SUPPORTED,
#                          minimizer_options=minimizer_options, tol=tol, verbosity=verbosity,
#                          maxiter=maxiter,
#                          minimize_func=ipopt.minimize_ipopt,
#                          strategy=strategy, criterion=criterion, name=name)
from ..settings import run
from ..util.exception import MaximumIterationReached


# import cyipopt


class IPoptV1(BaseMinimizer):
    _ALL_IPOPT_TOL = (
        'tiny_step_tol',  # xatol
        'tiny_step_y_tol',  # fatol
        # 'tiny_step_y_tol',  # fatol
    )

    def __init__(self,
                 tol: float = None,
                 verbosity: Optional[int] = None,
                 gradient: Optional[Union[Callable, str]] = None,
                 maxiter: Optional[Union[int, str]] = None,
                 options: Dict[str, object] = None,
                 hessian: Optional[str] = None,
                 maxcor: Optional[int] = None,
                 criterion: Optional[ConvergenceCriterion] = None,
                 strategy: Optional[ZfitStrategy] = None,
                 name="IPopt", ):
        """

        Args:
            tol (Union[float, None]):
            verbosity (int):
            gradient (Union[None, None]):
            hessian: Determine which hessian matrix to use during the minimization.
              One of the following option is possible
              - 'bfgs': BFGS quasi-Newton update formula for the limited approximation, update with skipping
              - 'sr1': SR1 quasi-Newton update formula for the limited approximation, update (doesn't work too well)
              - 'exact': Minimizer uses internally an exact calculation of the hessian using a numerical method.
              - 'zfit': use the exact hessian provided by the loss (either the automatic gradient or the numerical gradient
                computed inside the loss). This tends to be slow compared to the approximations and is usually
                not necessary.
            maxcor: Maximum number of memory history when using a quasi-Newton update formula

            options (): possible options for the minimizer. All options can be seen by using the command in the shell
                `ipopt --print_options`.
                A selection of parameters is presented here:
                - *alpha_red_factor*:  between 0 and 1, default 0.5
                    Fractional reduction of the trial step size in the backtracking line search.
                    At every step of the backtracking line search, the trial step size is
                    reduced by this factor.
                - *accept_after_max_steps*: -1 to +inf, default -1
                    Accept a trial point after maximal this number of steps.
                    Even if it does not satisfy line search conditions.

                - *watchdog_shortened_iter_trigger*: 0 to  +inf, default 10
                    Number of shortened iterations that trigger the watchdog.
                    If the number of successive iterations in which the backtracking line
                    search did not accept the first trial point exceeds this number, the
                    watchdog procedure is activated. Choosing "0" here disables the watchdog
                    procedure.
                - *watchdog_trial_iter_max*: 1 to +inf, default 3
                    Maximum number of watchdog iterations.
                    This option determines the number of trial iterations allowed before the
                    watchdog procedure is aborted and the algorithm returns to the stored
                    point.

                - *linear_solver*: default "mumps"
                    Linear solver used for step computations.
                    Determines which linear algebra package is to be used for the solution of
                    the augmented linear system (for obtaining the search directions). Note,
                    the code must have been compiled with the linear solver you want to
                    choose. Depending on your Ipopt installation, not all options are
                    available.
                    Possible values:
                    - ma27      [use the Harwell routine MA27]
                    - ma57      [use the Harwell routine MA57]
                    - ma77      [use the Harwell routine HSL_MA77]
                    - ma86      [use the Harwell routine HSL_MA86]
                    - ma97      [use the Harwell routine HSL_MA97]
                    - pardiso   [use the Pardiso package]
                    - wsmp      [use WSMP package]
                    - mumps     [use MUMPS package]
                    - custom    [use custom linear solver]
                - *mumps_pivtol*: ONLY FOR MUMPS
                    Pivot tolerance for the linear solver MUMPS.
                    A smaller number pivots for sparsity, a larger number pivots for
                    stability. This option is only available if Ipopt has been compiled with
                    MUMPS.


                - *mehrotra_algorithm*: default "no"
                    Indicates if we want to do Mehrotra's algorithm.
                    If set to yes, Ipopt runs as Mehrotra's predictor-corrector algorithm.
                    This works usually very well for LPs and convex QPs. This automatically
                    disables the line search, and chooses the (unglobalized) adaptive mu
                    strategy with the "probing" oracle, and uses "corrector_type=affine"
                    without any safeguards; you should not set any of those options
                    explicitly in addition. Also, unless otherwise specified, the values of
                    "bound_push", "bound_frac", and "bound_mult_init_val" are set more
                    aggressive, and sets "alpha_for_y=bound_mult".
                    Possible values:
                    - no      [Do the usual Ipopt algorithm.]
                    - yes     [Do Mehrotra's predictor-corrector algorithm.]

                - *fast_step_computation*: default "no"
                    Indicates if the linear system should be solved quickly.
                    If set to yes, the algorithm assumes that the linear system that is
                    solved to obtain the search direction, is solved sufficiently well. In
                    that case, no residuals are computed, and the computation of the search
                    direction is a little faster.
                    Possible values:
                    - no       [Verify solution of linear system by computing residuals.]
                    - yes      [Trust that linear systems are solved well.]

            maxiter (int):
            criterion (Union[None, None]):
            strategy (Union[None, None]):
            name (str):
        """
        minimizer_options = {}

        if gradient is None:
            gradient = 'zfit'
        if hessian is None:
            hessian = 'bfgs'
        options = {} if options is None else options
        minimizer_options['gradient'] = gradient
        minimizer_options['hessian'] = hessian
        if 'tol' in options:
            raise ValueError("Cannot put 'tol' into the options. Use `tol` in the init instead")
        if 'max_iter' in options:
            raise ValueError("Cannot put 'max_iter' into the options. Use `maxiter` instead.`")
        if 'limited_memory_update_type' in options:
            raise ValueError("Cannot put 'limited_memory_update_type' into the options."
                             " Use `hessian` instead.`")
        if 'limited_memory_max_history' in options:
            raise ValueError("Cannot put 'limited_memory_max_history' into the options."
                             " Use `numcor` instead.`")
        if 'hessian_approximation' in options:
            raise ValueError("Cannot put 'hessian_approximation' into the options."
                             " Use `hessian` instead.`")
        options['limited_memory_max_history'] = maxcor

        minimizer_options['ipopt'] = options

        internal_tol = {}
        for tol in self._ALL_IPOPT_TOL:
            if tol not in internal_tol:
                internal_tol[tol] = None
        self._internal_tol = internal_tol
        self._internal_maxiter = 20

        super().__init__(name=name,
                         tol=tol,
                         verbosity=verbosity,
                         minimizer_options=minimizer_options,
                         strategy=strategy,
                         criterion=criterion,
                         maxiter=maxiter)

    @minimize_supports(from_result=True)
    def _minimize(self, loss, params, init):
        previous_result = init
        evaluator = self.create_evaluator(loss, params)

        # initial values as array
        xvalues = np.array(run(params))

        # get and set the limits
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        nconstraints = 0
        empty_array = np.array([])
        nparams = len(params)
        hessian_sparsity_indices = np.meshgrid(range(nparams), range(nparams))

        def gradient_inplace(x, out):
            gradient = evaluator.gradient(x)
            out[:] = gradient

        minimizer_options = self.minimizer_options.copy()
        ipopt_options = minimizer_options.pop('ipopt').copy()
        print_level = 0
        if self.verbosity > 5:
            print_level = (self.verbosity - 5) * 2  # since the print-level is between 0 and 12
        if print_level == 10:
            print_level = 12
        if print_level == 9:
            print_level = 10
        ipopt_options['print_level'] = print_level
        if print_level == 12:
            ipopt_options['print_timing_statistics'] = 'yes'

        ipopt_options['tol'] = self.tol
        ipopt_options['max_iter'] = self.get_maxiter(len(params))
        hessian = minimizer_options['hessian']

        minimizer_kwargs = dict(n=nparams,
                                xL=lower, xU=upper,
                                m=nconstraints,
                                gL=empty_array, gU=empty_array,  # no constraints
                                sparsity_indices_jac_g=(empty_array, empty_array),
                                sparsity_indices_hess=hessian_sparsity_indices,
                                eval_f=evaluator.value,
                                eval_grad_f=gradient_inplace,
                                eval_g=lambda x, out: None,
                                eval_jac_g=lambda x, out: None)
        if hessian == 'zfit':
            def hessian_inplace(x, out):
                hessian = evaluator.hessian(x)
                out[:] = hessian

            minimizer_kwargs['eval_h'] = hessian_inplace
        else:
            if hessian == 'exact':
                ipopt_options['hessian_approximation'] = hessian

            else:
                ipopt_options['hessian_approximation'] = 'limited-memory'
                ipopt_options['limited_memory_update_type'] = hessian

        # ipopt_options['dual_inf_tol'] = TODO?

        minimizer = ipyopt.Problem(**minimizer_kwargs)

        minimizer.set(**{k: v for k, v in ipopt_options.items() if v is not None})

        criterion = self.criterion(tol=self.tol, loss=loss, params=params)

        init_tol = min([math.sqrt(loss.errordef * self.tol), loss.errordef * self.tol * 1e2])
        # init_tol **= 0.5
        internal_tol = self._internal_tol
        internal_tol = {tol: init_tol if init is None else init for tol, init in internal_tol.items()}

        valid = True
        edm = None
        criterion_value = None
        valid_message = ""

        warm_start_options = (
            'warm_start_init_point',
            # 'warm_start_same_structure',
            'warm_start_entire_iterate'
        )
        minimizer.set_intermediate_callback(lambda *a, **k: print(a, k) or True)

        fmin = -999
        status = -999
        converged = False
        for i in range(self._internal_maxiter):

            minimizer.set(**internal_tol)

            # run the minimization
            try:
                xvalues, fmin, status = minimizer.solve(xvalues,
                                                        # mult_g=constraint_multipliers,
                                                        # mult_x_L=zl,
                                                        # mult_x_U=zu
                                                        )
            except MaximumIterationReached:
                maxiter_reached = True
                valid = False
                valid_message = "Maxiter reached, terminated without convergence"
            else:
                maxiter_reached = evaluator.niter > evaluator.maxiter

            set_values(params, xvalues)
            with evaluator.ignore_maxiter():
                result_prelim = FitResult.from_ipopt(loss=loss,
                                                     params=params,
                                                     xvalues=xvalues,
                                                     minimizer=self,
                                                     opt_instance=minimizer,
                                                     fmin=fmin,
                                                     converged=converged,
                                                     status=status,
                                                     edm=CRITERION_NOT_AVAILABLE,
                                                     evaluator=evaluator,
                                                     valid=valid,
                                                     niter=None,
                                                     criterion=criterion,
                                                     message=valid_message)
                converged = criterion.converged(result_prelim) and valid
            criterion_value = criterion.last_value
            if isinstance(criterion, EDM):
                edm = criterion.last_value
            else:
                edm = CRITERION_NOT_AVAILABLE

            if self.verbosity > 5:
                print_minimization_status(converged=converged, criterion=criterion, evaluator=evaluator,
                                          i=i,
                                          fmin=fmin,
                                          internal_tol=internal_tol)

            if converged or maxiter_reached:
                break

            # prepare for next run
            minimizer.set(**{option: 'yes' for option in warm_start_options})

            # update the tolerances
            self._update_tol_inplace(criterion_value=criterion_value, internal_tol=internal_tol)

        else:
            valid = False
            valid_message = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tol} not reached."

        # cleanup of convergence
        minimizer.set(**{option: 'no' for option in warm_start_options})

        return FitResult.from_ipopt(loss=loss,
                                    params=params,
                                    minimizer=self,
                                    xvalues=xvalues,
                                    opt_instance=minimizer,
                                    fmin=fmin,
                                    status=status,
                                    edm=edm,
                                    criterion=criterion,
                                    niter=None,
                                    converged=converged,
                                    evaluator=evaluator,
                                    valid=valid,
                                    message=valid_message)

#  Copyright (c) 2023 zfit

from __future__ import annotations

import math

import numpy as np

from .baseminimizer import BaseMinimizer, minimize_supports, print_minimization_status
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import CRITERION_NOT_AVAILABLE, EDM, ConvergenceCriterion
from ..core.parameter import assign_values
from ..settings import run
from ..util.exception import MaximumIterationReached


class IpyoptV1(BaseMinimizer):
    _ALL_IPOPT_TOL = (
        "tiny_step_tol",  # xatol
        "tiny_step_y_tol",  # fatol
        # 'tiny_step_y_tol',  # fatol
    )

    def __init__(
        self,
        tol: float | None = None,
        maxcor: int | None = None,
        verbosity: int | None = None,
        hessian: str | None = None,
        options: dict[str, object] | None = None,
        maxiter: int | str | None = None,
        criterion: ConvergenceCriterion | None = None,
        strategy: ZfitStrategy | None = None,
        name: str | None = "IpyoptV1",
    ) -> None:
        """Ipopt is a gradient-based minimizer that performs large scale nonlinear optimization of continuous systems.

        This implemenation uses the `IPyOpt wrapper <https://gitlab.com/g-braeunlich/ipyopt>`_

         `Ipopt <https://coin-or.github.io/Ipopt/index.html>`_
         (Interior Point Optimizer, pronounced "Eye-Pea-Opt") is an open source software package for
         large-scale nonlinear optimization. It can be used to solve general nonlinear programming problems
         It is written in Fortran and C and is released under the EPL (formerly CPL).
         IPOPT implements a primal-dual interior point method, and uses line searches based on
         Filter methods (Fletcher and Leyffer).

        IPOPT is part of the `COIN-OR <https://www.coin-or.org/>`_ project.

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            maxcor: |@doc:minimizer.maxcor| Maximum number of memory history to keep
                   when using a quasi-Newton update formula such as BFGS.
                   It is the number of gradients
                   to “remember” from previous optimization
                   steps: increasing it increases
                   the memory requirements but may speed up the convergence. |@docend:minimizer.maxcor|
            verbosity: |@doc:minimizer.verbosity| Verbosity of the minimizer. Has to be between 0 and 10.
              The verbosity has the meaning:

               - a value of 0 means quiet and no output
               - above 0 up to 5, information that is good to know but without
                 flooding the user, corresponding to a "INFO" level.
               - A value above 5 starts printing out considerably more and
                 is used more for debugging purposes.
               - Setting the verbosity to 10 will print out every
                 evaluation of the loss function and gradient.

               Some minimizers offer additional output which is also
               distributed as above but may duplicate certain printed values. |@docend:minimizer.verbosity|
            hessian: Determine which hessian matrix to use during the minimization.
              One of the following option is possible

              - 'bfgs': BFGS quasi-Newton update formula for the limited approximation, update with skipping
              - 'sr1': SR1 quasi-Newton update formula for the limited approximation, update (doesn't work too well)
              - 'exact': Minimizer uses internally an exact calculation of the hessian using a numerical method.
              - 'zfit': use the exact hessian provided by the loss (either the automatic gradient or the numerical gradient
                computed inside the loss). This tends to be slow compared to the approximations and is usually
                not necessary.

            options: Additional possible options for the minimizer. All options can be seen by using
            the command in the shell

            .. code-block:: bash

                `ipopt --print-options`

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

            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        minimizer_options = {}

        if hessian is None:
            hessian = "bfgs"
        options = {} if options is None else options
        minimizer_options["hessian"] = hessian
        if "tol" in options:
            raise ValueError(
                "Cannot put 'tol' into the options. Use `tol` in the init instead"
            )
        if "max_iter" in options:
            raise ValueError(
                "Cannot put 'max_iter' into the options. Use `maxiter` instead.`"
            )
        if "limited_memory_update_type" in options:
            raise ValueError(
                "Cannot put 'limited_memory_update_type' into the options."
                " Use `hessian` instead.`"
            )
        if "limited_memory_max_history" in options:
            raise ValueError(
                "Cannot put 'limited_memory_max_history' into the options."
                " Use `numcor` instead.`"
            )
        if "hessian_approximation" in options:
            raise ValueError(
                "Cannot put 'hessian_approximation' into the options."
                " Use `hessian` instead.`"
            )
        options["limited_memory_max_history"] = maxcor

        minimizer_options["ipopt"] = options

        internal_tol = {}
        for iptol in self._ALL_IPOPT_TOL:
            if iptol not in internal_tol:
                internal_tol[iptol] = None
        self._internal_tol = internal_tol
        self._internal_maxiter = 20

        try:
            import ipyopt
        except ImportError as error:
            raise ImportError(
                "This requires the ipyopt library (https://gitlab.com/g-braeunlich/ipyopt)"
                " to be installed. On a 'Linux' environment, you can install zfit with"
                " `pip install zfit[ipyopt]` (or install ipyopt with pip). For MacOS, there are currently"
                " no wheels (but will come in the future). In this case, please install ipyopt manually "
                "to use this minimizer"
                " or install zfit on a 'Linux' environment."
            ) from error

        super().__init__(
            name=name,
            tol=tol,
            verbosity=verbosity,
            minimizer_options=minimizer_options,
            strategy=strategy,
            criterion=criterion,
            maxiter=maxiter,
        )

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init):
        import ipyopt

        if init:
            assign_values(params=params, values=init)
        evaluator = self.create_evaluator(numpy_converter=np.array)
        criterion = self.create_criterion()

        # initial values as array
        xvalues = np.array(run(params))

        # get and set the limits
        lower = np.array([p.lower for p in params])
        upper = np.array([p.upper for p in params])
        nconstraints = 0
        empty_array = np.array([])
        nparams = len(params)
        hessian_sparsity_indices = np.meshgrid(range(nparams), range(nparams))

        minimizer_options = self.minimizer_options.copy()

        def gradient_inplace(x, out):
            gradient = evaluator.gradient(x)
            out[:] = gradient

        ipopt_options = minimizer_options.pop("ipopt").copy()
        print_level = self.verbosity
        if print_level == 8:
            print_level = 9
        elif print_level == 9:
            print_level = 11
        elif print_level == 10:
            if "print_timing_statistics" not in ipopt_options:
                ipopt_options["print_timing_statistics"] = "yes"
        ipopt_options["print_level"] = print_level

        ipopt_options["tol"] = self.tol
        ipopt_options["max_iter"] = self.get_maxiter()
        hessian = minimizer_options.pop("hessian")

        minimizer_kwargs = dict(
            n=nparams,
            x_l=lower,
            x_u=upper,
            m=nconstraints,
            g_l=empty_array,
            g_u=empty_array,  # no constraints
            sparsity_indices_jac_g=(empty_array, empty_array),
            sparsity_indices_h=hessian_sparsity_indices,
            eval_f=evaluator.value,
            eval_grad_f=gradient_inplace,
            eval_g=lambda x, out: None,
            eval_jac_g=lambda x, out: None,
        )
        if hessian == "zfit":

            def hessian_inplace(x, out):
                hessian = evaluator.hessian(x)
                out[:] = hessian

            minimizer_kwargs["eval_h"] = hessian_inplace
        else:
            if hessian == "exact":
                ipopt_options["hessian_approximation"] = hessian

            else:
                ipopt_options["hessian_approximation"] = "limited-memory"
                ipopt_options["limited_memory_update_type"] = hessian
        # ipopt_options['dual_inf_tol'] = TODO?

        minimizer = ipyopt.Problem(**minimizer_kwargs)

        minimizer.set(**{k: v for k, v in ipopt_options.items() if v is not None})

        init_tol = min(
            [math.sqrt(loss.errordef * self.tol), loss.errordef * self.tol * 1e2]
        )
        # init_tol **= 0.5
        internal_tol = self._internal_tol
        internal_tol = {
            tol: init_tol if init is None else init
            for tol, init in internal_tol.items()
        }

        valid = True
        edm = None
        criterion_value = None
        valid_message = ""

        warm_start_options = (  # TODO: what exactly here?
            # "warm_start_init_point",
            # 'warm_start_same_structure',
            "warm_start_entire_iterate",
        )
        # minimizer.set_intermediate_callback(lambda *a, **k: print(a, k) or True)

        fmin = None
        status = None
        converged = False
        for i in range(self._internal_maxiter):
            minimizer.set(**internal_tol)

            # run the minimization
            try:
                xvalues, fmin, status = minimizer.solve(
                    xvalues,
                    # mult_g=constraint_multipliers,
                    # mult_x_L=zl,
                    # mult_x_U=zu
                )
            except MaximumIterationReached:
                maxiter_reached = True
                valid = False
                valid_message = "Maxiter reached, terminated without convergence"
            else:
                maxiter_reached = evaluator.maxiter_reached

            assign_values(params, xvalues)
            with evaluator.ignore_maxiter():
                result_prelim = FitResult.from_ipopt(
                    loss=loss,
                    params=params,
                    values=xvalues,
                    minimizer=self,
                    problem=minimizer,
                    fmin=fmin,
                    converged=converged,
                    status=status,
                    edm=CRITERION_NOT_AVAILABLE,
                    evaluator=evaluator,
                    valid=valid,
                    niter=None,
                    criterion=criterion,
                    message=valid_message,
                )
                converged = criterion.converged(result_prelim)
            criterion_value = criterion.last_value
            if isinstance(criterion, EDM):
                edm = criterion.last_value
            else:
                edm = CRITERION_NOT_AVAILABLE

            if self.verbosity > 5:
                print_minimization_status(
                    converged=converged,
                    criterion=criterion,
                    evaluator=evaluator,
                    i=i,
                    fmin=fmin,
                    internal_tol=internal_tol,
                )

            if converged or maxiter_reached:
                break

            # prepare for next run
            minimizer.set(**{option: "yes" for option in warm_start_options})

            # update the tolerances
            self._update_tol_inplace(
                criterion_value=criterion_value, internal_tol=internal_tol
            )

        else:
            valid = False
            valid_message = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tol} not reached."

        # cleanup of convergence
        minimizer.set(**{option: "no" for option in warm_start_options})
        assign_values(params=params, values=xvalues)

        return FitResult.from_ipopt(
            loss=loss,
            params=params,
            minimizer=self,
            values=xvalues,
            problem=minimizer,
            fmin=fmin,
            status=status,
            edm=edm,
            criterion=criterion,
            niter=None,
            converged=converged,
            evaluator=evaluator,
            valid=valid,
            message=valid_message,
        )

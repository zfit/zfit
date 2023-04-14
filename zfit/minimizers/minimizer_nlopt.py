#  Copyright (c) 2023 zfit

from __future__ import annotations

import collections
import copy
import math
from collections.abc import Mapping, Callable

from ..util.checks import RuntimeDependency

try:
    import nlopt
except ImportError:
    nlopt = RuntimeDependency("nlopt")

import numpy as np

from ..core.parameter import assign_values
from ..settings import run
from ..util.exception import MaximumIterationReached
from .baseminimizer import (
    NOT_SUPPORTED,
    BaseMinimizer,
    minimize_supports,
    print_minimization_status,
)
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import CRITERION_NOT_AVAILABLE, EDM, ConvergenceCriterion


class NLoptBaseMinimizerV1(BaseMinimizer):
    _ALL_NLOPT_TOL = (
        # 'fatol',
        "ftol",
        "xatol",
        "xtol",
    )

    def __init__(
        self,
        algorithm: int,
        tol: float | None = None,
        gradient: Callable | str | NOT_SUPPORTED | None = NOT_SUPPORTED,
        hessian: Callable | str | NOT_SUPPORTED | None = NOT_SUPPORTED,
        maxiter: int | str | None = None,
        minimizer_options: Mapping[str, object] | None = None,
        internal_tols: Mapping[str, float | None] = None,
        verbosity: int | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt Base Minimizer V1",
    ):
        """NLopt is a library that contains multiple different optimization algorithms.

         |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

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

             tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
             gradient: Gradient that will be given to the minimizer if supported.
             hessian: Hessian that will be given to the minimizer if supported.
             internal_tols: Tolerances for the minimizer. Has to contain possible tolerance criteria.
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
             maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
             minimizer_options: Additional options that will be set in the minimizer.
             name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        try:
            import nlopt
        except ImportError:
            raise ImportError(
                "nlopt is not installed. This is an optional dependency. To include it,you"
                " can install zfit with `pip install zfit[nlopt]` or `pip install zfit[all]`."
            )
        self._algorithm = algorithm
        if minimizer_options is None:
            minimizer_options = {}
        minimizer_options = copy.copy(minimizer_options)

        if gradient is not NOT_SUPPORTED:
            if gradient is False:
                raise ValueError("grad cannot be False for NLopt minimizer.")
            minimizer_options["gradient"] = gradient
        if hessian is not NOT_SUPPORTED:
            minimizer_options["hessian"] = hessian

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
        previous_result = init
        if init:
            assign_values(params=params, values=init)
        evaluator = self.create_evaluator(loss, params)

        # create minimizer instance
        minimizer = nlopt.opt(nlopt.LD_LBFGS, len(params))

        # initial values as array
        xvalues = initial_xvalues = np.asarray(run(params))

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
                grad[:] = np.asarray(gradients)
            else:
                value = evaluator.value(x)

            return float(value)

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

        maxcor = minimizer_options.pop("maxcor", None)
        if maxcor is not None:
            minimizer.set_vector_storage(maxcor)

        population = minimizer_options.pop("population", None)
        if population is not None:
            minimizer.set_population(population)

        for name, value in minimizer_options.items():
            minimizer.set_param(name, value)

        minimizer.set_param("verbosity", max(0, self.verbosity - 6))

        criterion = self.criterion(tol=self.tol, loss=loss, params=params)
        init_tol = min(
            [math.sqrt(loss.errordef * self.tol), loss.errordef * self.tol * 1e3]
        )
        # init_tol *= 10
        internal_tol = self._internal_tols
        internal_tol = {
            tol: init_tol if init is None else init
            for tol, init in internal_tol.items()
        }
        if "xtol" in internal_tol:
            internal_tol["xtol"] **= 0.5
        if "ftol" in internal_tol:
            internal_tol["ftol"] **= 0.5

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
                approx_step_sizes = result_prelim.hesse(
                    params=params, method="approx", name="approx"
                )
            empty_dict = {}
            for param in params:
                step_size = approx_step_sizes.get(param, empty_dict).get("error")
                if step_size is None and param.has_step_size:
                    step_size = param.step_size
                init_scale.append(step_size)
            minimizer.set_initial_step(init_scale)

            self._set_tols_inplace(
                minimizer=minimizer,
                internal_tol=internal_tol,
                criterion_value=criterion_value,
            )

            # some (global) optimizers use a local minimizer, set that here
            if local_minimizer is not None:
                self._set_tols_inplace(
                    minimizer=local_minimizer,
                    internal_tol=internal_tol,
                    criterion_value=criterion_value,
                )

                minimizer.set_local_optimizer(local_minimizer)

            # run the minimization
            try:
                xvalues = minimizer.optimize(xvalues)
            except MaximumIterationReached:
                maxiter_reached = True
                valid = False
                valid_message = "Maxiter reached, terminated without convergence"
            except RuntimeError:
                if self.verbosity > 3:
                    print(
                        "Minimization in NLopt failed, restarting with slightly varied parameters."
                    )
                if nrandom < self._nrandom_max:  # in order not to start too close
                    init_scale_isnot_none = np.asarray(
                        [scale is not None for scale in init_scale], dtype=bool
                    )
                    init_scale = np.where(
                        init_scale_isnot_none,
                        init_scale,
                        np.ones_like(init_scale, dtype=np.float64),
                    )

                    init_scale_no_nan = np.nan_to_num(init_scale, nan=1.0)
                    init_scale_no_nan = init_scale_no_nan.astype(np.float64)
                    upper_random = np.minimum(
                        initial_xvalues + init_scale_no_nan / 2, upper
                    )
                    lower_random = np.maximum(
                        initial_xvalues - init_scale_no_nan / 2, lower
                    )
                    initial_xvalues = np.random.uniform(
                        low=lower_random, high=upper_random
                    )

                    nrandom += 1
            else:
                maxiter_reached = evaluator.niter > evaluator.maxiter

            assign_values(params, xvalues)
            fmin = (
                minimizer.last_optimum_value()
            )  # TODO: what happens if minimization terminated?
            with evaluator.ignore_maxiter():
                result_prelim = FitResult.from_nlopt(
                    loss,
                    minimizer=self,
                    opt=minimizer,
                    edm=CRITERION_NOT_AVAILABLE,
                    niter=evaluator.nfunc_eval,
                    params=params,
                    evaluator=evaluator,
                    criterion=None,
                    values=xvalues,
                    valid=valid,
                    message=valid_message,
                )
                converged = criterion.converged(result_prelim)
                valid = converged
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

            # update the tols
            self._update_tol_inplace(
                criterion_value=criterion_value, internal_tol=internal_tol
            )

        else:
            valid = False
            valid_message = f"Invalid, criterion {criterion.name} is {criterion_value}, target {self.tol} not reached."

        return FitResult.from_nlopt(
            loss,
            minimizer=self.copy(),
            opt=minimizer,
            edm=edm,
            niter=evaluator.niter,
            params=params,
            evaluator=evaluator,
            values=xvalues,
            valid=valid,
            criterion=criterion,
            message=valid_message,
        )

    def _set_tols_inplace(self, minimizer, internal_tol, criterion_value):
        # set all the tolerances
        fatol = internal_tol.get("fatol")
        if fatol is not None:
            minimizer.set_ftol_abs(fatol**0.5)
        xatol = internal_tol.get("xatol")
        if xatol is not None:
            # minimizer.set_xtol_abs([xatol] * len(params))
            minimizer.set_xtol_abs(xatol)
        # set relative tolerances later as it can be unstable. Just use them when approaching
        if criterion_value is not None:
            tol_factor_full = self.tol / criterion_value
            if tol_factor_full < 1e-8:
                ftol = internal_tol.get("ftol")
                if ftol is not None:
                    minimizer.set_ftol_rel(ftol)

                xtol = internal_tol.get("xtol")
                if xtol is not None:
                    # minimizer.set_xtol_rel([xtol] * len(params))  # TODO: one value or vector?
                    minimizer.set_xtol_rel(xtol)  # TODO: one value or vector?


class NLoptLBFGSV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        maxcor: int | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt L-BFGS V1",
    ) -> None:
        """Local, gradient-based quasi-Newton minimizer using the low storage BFGS Hessian approximation.

        This is most probably the most popular algorithm for gradient based local minimum searches and also
        the underlying algorithm in the
        `Minuit <https://www.sciencedirect.com/science/article/abs/pii/0010465575900399>`_ minimizer that is
        also available as :class:`~zfit.minimize.Minuit`.


        This algorithm is based on a Fortran implementation of the low-storage BFGS algorithm
        written by Prof. Ladislav Luksan, and graciously posted online under the GNU LGPL at:

        -   <http://www.uivt.cas.cz/~luksan/subroutines.html>

        The original L-BFGS algorithm, based on variable-metric updates via Strang recurrences,
         was described by the papers:

        -   J. Nocedal, "Updating quasi-Newton matrices with limited storage," *Math. Comput.* **35**, 773-782 (1980).
        -   D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization,"
            ''Math. Programming' **45**, p. 503-528 (1989).

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LD_LBFGS,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={"maxcor": maxcor},
            strategy=strategy,
            maxiter=maxiter,
        )

        @property
        def maxcor(self):
            return self.minimizer_options.get("maxcor")


class NLoptShiftVarV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        maxcor: int | None = None,
        rank: int | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name="NLopt Shifted Variable Memory",
    ) -> None:
        """Local, gradient-based minimizer using a shifted limited-memory variable-metric.

        This algorithm is based on a Fortran implementation of a shifted limited-memory variable-metric
        algorithm by Prof. Ladislav Luksan, and graciously posted online under the GNU LGPL at:

        -   <http://www.uivt.cas.cz/~luksan/subroutines.html>

        There are two variations of this algorithm: either using a rank-2 method or a rank-1 method.

        The algorithms are based on the ones described by:

        -   J. Vlcek and L. Luksan, "Shifted limited-memory variable metric methods for large-scale unconstrained
            minimization," *J. Computational Appl. Math.* **186**, p. 365-390 (2006).

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

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
            rank: Rank of the algorithm used, either 1 or 2. Defaults to 2.
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """

        if rank is None:
            rank = 2

        if rank == 1:
            algorithm = nlopt.LD_VAR1
        elif rank == 2:
            algorithm = nlopt.LD_VAR2
        else:
            raise ValueError(f"rank has to be either 1 or 2, not {rank}")
        self._rank = rank
        super().__init__(
            name=name,
            algorithm=algorithm,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={"maxcor": maxcor},
            strategy=strategy,
            maxiter=maxiter,
        )

        @property
        def rank(self):
            return self._rank

        @property
        def maxcor(self):
            return self.minimizer_options.get("maxcor")


class NLoptTruncNewtonV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        maxcor: int | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name="NLopt Truncated Newton",
    ) -> None:
        """Local, gradient-based truncated Newton minimizer using an inexact algorithm.

        This algorithm is based on a Fortran implementation of a
        preconditioned inexact truncated Newton algorithm written by
        Prof. Ladislav Luksan, and graciously posted online under the GNU LGPL
        at:

        -  http://www.uivt.cas.cz/~luksan/subroutines.html

        The algorithms is based on the ones described by:

        -  R. S. Dembo and T. Steihaug, “Truncated Newton algorithms for
           large-scale optimization,” *Math. Programming* **26**, p. 190-212
           (1983) http://doi.org/10.1007/BF02592055.

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LD_TNEWTON_PRECOND_RESTART,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={"maxcor": maxcor},
            strategy=strategy,
            maxiter=maxiter,
        )

        @property
        def maxcor(self):
            return self.minimizer_options.get("maxcor")


class NLoptSLSQPV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt SLSQP",
    ) -> None:
        r"""Local gradient based minimizer using a sequential quadratic programming.

        This is a sequential quadratic programming (SQP) algorithm for
        non-linearly gradient-based optimization based on the implementation by
        Dieter Kraft and described in:

        -  Dieter Kraft, “A software package for sequential quadratic
           programming”, Technical Report DFVLR-FB 88-28, Institut für Dynamik
           der Flugsysteme, Oberpfaffenhofen, July 1988.
        -  Dieter Kraft, “Algorithm 733: TOMP–Fortran modules for optimal
           control calculations,” *ACM Transactions on Mathematical Software*,
           vol. 20, no. 3, pp. 262-281 (1994).

        The algorithm optimizes
        successive second-order (quadratic/least-squares) approximations of the
        objective function (via BFGS updates), with first-order (affine)
        approximations of the constraints.

        The Fortran code was obtained from the SciPy project, who are
        responsible for `obtaining permission`_ to distribute it under a
        free-software (3-clause BSD) license.

        The code was modified for inclusion in NLopt by S. G. Johnson in 2010,
        with the following changes. The code was converted to C and manually
        cleaned up. It was modified to be re-entrant (preserving the
        reverse-communication interface but explicitly saving the state in a
        data structure). The inexact line
        search was modified to evaluate the functions including gradients for
        the first step, since this removes the need to evaluate the
        function+gradient a second time for the same point in the common case
        when the inexact line search concludes after a single step.

        **Note:** Because the SLSQP code uses dense-matrix methods (ordinary
        BFGS, not low-storage BFGS), it requires *O*\ (*n*\ 2) storage and
        *O*\ (*n*\ 3) time in *n* dimensions, which makes it less practical for
        optimizing more than a few thousand parameters.

        .. _obtaining permission: http://permalink.gmane.org/gmane.comp.python.scientific.devel/6725

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LD_SLSQP,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptBOBYQAV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt BOBYQA",
    ) -> None:
        """Derivative-free local minimizer that iteratively constructed quadratic approximation for the loss.

        This is an algorithm derived from the `BOBYQA subroutine`_ of M. J. D.
        Powell, converted to C and modified for the NLopt stopping criteria.
        BOBYQA performs derivative-free bound-constrained optimization using an
        iteratively constructed quadratic approximation for the objective
        function. See:

        -  M. J. D. Powell, “`The BOBYQA algorithm for bound constrained
           optimization without derivatives`_,” Department of Applied
           Mathematics and Theoretical Physics, Cambridge England, technical
           report NA2009/06 (2009).

        (Because BOBYQA constructs a quadratic approximation of the objective,
        it may perform poorly for objective functions that are not
        twice-differentiable.)

        This algorithm largely
        supersedes the NEWUOA algorithm, which is an earlier version of
        the same idea by Powell.

        .. _BOBYQA subroutine: http://plato.asu.edu/ftp/other_software/bobyqa.zip
        .. _The BOBYQA algorithm for bound constrained optimization without derivatives: http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LN_BOBYQA,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptMMAV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt MMA",
    ):
        """Method-of-moving-asymptotes for gradient-based local minimization.

        Globally-convergent method-of-moving-asymptotes (MMA) for gradient-based local minimization.
        The algorithm is described in:

        -  Krister Svanberg, “`A class of globally convergent optimization
           methods based on conservative convex separable approximations`_,”
           *SIAM J. Optim.* **12** (2), p. 555-573 (2002).

        This is an improved CCSA (“conservative convex separable approximation”)
        variant of the original MMA algorithm published by Svanberg in 1987,
        which has become popular for topology optimization. (*Note:* “globally
        convergent” does *not* mean that this algorithm converges to the global
        optimum; it means that it is guaranteed to converge to *some* local
        minimum from any feasible starting point.)

        At each point **x**, MMA forms a local approximation using the gradient
        of *f* and the constraint functions, plus a quadratic “penalty” term to
        make the approximations “conservative” (upper bounds for the exact
        functions). The precise approximation MMA forms is difficult to describe
        in a few words, because it includes nonlinear terms consisting of a
        poles at some distance from *x* (outside of the current trust region),
        almost a kind of Padé approximant. The main point is that the
        approximation is both convex and separable, making it trivial to solve
        the approximate optimization by a dual method. Optimizing the
        approximation leads to a new candidate point **x**. The objective and
        constraints are evaluated at the candidate point. If the approximations
        were indeed conservative (upper bounds for the actual functions at the
        candidate point), then the process is restarted at the new **x**.
        Otherwise, the approximations are made more conservative (by increasing
        the penalty term) and re-optimized.


        .. _A class of globally convergent optimization methods based on conservative convex separable approximations: http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.146.5196
        .. _Professor Svanberg: http://researchprojects.kth.se/index.php/kb_7902/pb_2085/pb.html

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LD_MMA,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptCCSAQV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt CCSAQ",
    ):
        """MMA-like minimizer with simpler, quadratic local approximations.

        CCSA is an algorithm from the same paper as MMA as described in:

        -  Krister Svanberg, “`A class of globally convergent optimization
           methods based on conservative convex separable approximations`_,”
           *SIAM J. Optim.* **12** (2), p. 555-573 (2002).

        CCSA has the following differences:
        instead of constructing local MMA approximations, it
        constructs simple quadratic approximations (or rather, affine
        approximations plus a quadratic penalty term to stay conservative). This
        is the ccsa_quadratic code. It seems to have similar convergence rates
        to MMA for most problems, which is not surprising as they are both
        essentially similar.

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LD_CCSAQ,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptCOBYLAV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt COBYLA",
    ):
        r"""Derivative free simplex minimizer using a linear approximation with trust region steps.

        COBYLA (Constrained Optimization BY Linear Approximations) constructs successive linear approximations of the
        objective function and constraints via a simplex of n+1 points (in n dimensions), and optimizes these
        approximations in a trust region at each step.

        This is a derivative of Powell’s implementation of the COBYLA  algorithm for derivative-free optimization
        by M. J. D. Powell described in:

        -  M. J. D. Powell, “A direct search optimization method that models the
           objective and constraint functions by linear interpolation,” in
           *Advances in Optimization and Numerical Analysis*, eds. S. Gomez and
           J.-P. Hennart (Kluwer Academic: Dordrecht, 1994), p. 51-67.

        and reviewed in:

        -  M. J. D. Powell, “Direct search algorithms for optimization
           calculations,” *Acta Numerica* **7**, 287-336 (1998).

        It constructs successive linear approximations of the objective function
        and constraints via a simplex of *n*\ +1 points (in *n* dimensions), and
        optimizes these approximations in a trust region at each step.

        The original code itself was written in Fortran by Powell and was
        converted to C in 2004 by Jean-Sebastien Roy (js@jeannot.org) for the
        SciPy project. The version in NLopt was based on Roy’s C version and offers a few improvements
        over the original code:

        - `COBYLA` can increase the trust-region radius if the predicted improvement
          was approximately right and the simplex is OK
        - pseudo-randomization of the simplex steps in the COBYLA algorithm improve the robustness by avoiding
           accidentally taking steps that don't improve conditioning.

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|


        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LN_COBYLA,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptSubplexV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt Subplex",
    ):
        """Local derivative free minimizer which improves on the Nealder-Mead algorithm.

        This is a re-implementation of Tom Rowan’s “Subplex” algorithm.

        Subplex (a variant of Nelder-Mead that uses Nelder-Mead on a sequence of
        subspaces) is claimed to be much more efficient and robust than the
        original Nelder-Mead, while retaining the latter’s facility with
        discontinuous objectives.

        The description of Rowan’s algorithm in his PhD thesis is used:

        -  T. Rowan, “Functional Stability Analysis of Numerical Algorithms”,
           Ph.D. thesis, Department of Computer Sciences, University of Texas at
           Austin, 1990.

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|


        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        super().__init__(
            name=name,
            algorithm=nlopt.LN_SBPLX,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options={},
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptMLSLV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        population: int | None = None,
        randomized: bool | None = None,
        local_minimizer: int | Mapping[str, object] | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt MLSL",
    ) -> None:
        """Global minimizer using local optimization by randomly selecting points.

        “Multi-Level Single-Linkage” (MLSL) is an algorithm for global optimization by
        a sequence of local optimizations from random starting points, proposed
        by:

        -  A. H. G. Rinnooy Kan and G. T. Timmer, “Stochastic global
           optimization methods,” *Mathematical Programming*, vol. 39, p. 27-78
           (1987). (Actually 2 papers — part I: clustering methods, p. 27, then
           part II: multilevel methods, p. 57.)

        We also include a modification of MLSL use a Sobol’ `low-discrepancy
        sequence`_ (LDS), also used in so-called
        `quasi Monte Carlo methods <https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method>`_
        that can be invoked by setting *randomized* to False
        (as it is now *quasi*randomized) instead of pseudorandom numbers, which was argued to
        improve the convergence rate by:

        -  Sergei Kucherenko and Yury Sytsko, “Application of deterministic
           low-discrepancy sequences in global optimization,” *Computational
           Optimization and Applications*, vol. 30, p. 297-318 (2005).

        In either case, MLSL is a “multistart” algorithm: it works by doing a
        sequence of local optimizations (using some other local optimization
        algorithm) from random or low-discrepancy starting points. MLSL is
        distinguished, however by a “clustering” heuristic that helps it to
        avoid repeated searches of the same local optima, and has some
        theoretical guarantees of finding all local optima in a finite number of
        local minimizations.

        The local-search portion of MLSL can use any of the other algorithms in
        NLopt. The local search uses the
        derivative/nonderivative algorithm set by the *local_minimizer* argument.


        .. _low-discrepancy sequence: https://en.wikipedia.org/wiki/Low-discrepancy_sequence
        .. _local optimization: NLopt_Reference#localsubsidiary-optimization-algorithm

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            population: |@doc:minimizer.nlopt.population| The population size for the evolutionary algorithm. |@docend:minimizer.nlopt.population|
                By default, each iteration of MLSL samples 4 random new trial points.
            randomized: If True, uses the randomized version 'GD_MLSL_LDS' instead of 'GD_MLSL'
            local_minimizer: Configuration for the local minimizer. Defaults to L-BFGS.
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """
        if randomized is None:
            randomized = False
        if randomized:
            algorithm = nlopt.GD_MLSL
        else:
            algorithm = nlopt.GD_MLSL_LDS

        local_minimizer = nlopt.LD_LBFGS if local_minimizer is None else local_minimizer
        if not isinstance(local_minimizer, collections.abc.Mapping):
            local_minimizer = {"algorithm": local_minimizer}
        if "algorithm" not in local_minimizer:
            raise ValueError("algorithm needs to be specified in local_minimizer")

        minimizer_options = {"local_minimizer_options": local_minimizer}
        if population is not None:
            minimizer_options["population"] = population
        super().__init__(
            name=name,
            algorithm=algorithm,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options=minimizer_options,
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptStoGOV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        randomized: bool | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt MLSL",
    ):
        """Global minimizer which divides the space into smaller rectangles and uses a local BFGS variant inside.

        StoGO is a global optimization algorithm that works by systematically dividing
        the search space (which must be bound-constrained) into smaller
        hyper-rectangles via a branch-and-bound technique, and searching them by
        a gradient-based local-search algorithm (a BFGS variant), optionally
        including some randomness (hence the “Sto”, which stands for
        “stochastic” I believe).

        Some references on StoGO are:

        -  S. Gudmundsson, “Parallel Global Optimization,” M.Sc. Thesis, IMM,
           Technical University of Denmark, 1998.
        -  K. Madsen, S. Zertchaninov, and A. Zilinskas, “Global Optimization
           using Branch-and-Bound,” unpublished (1998). A preprint of this paper
           is included in the ``stogo`` subdirectory of NLopt as ``paper.pdf``.
        -  S. Zertchaninov and K. Madsen, “A C++ Programme for Global
           Optimization,” IMM-REP-1998-04, Department of Mathematical Modelling,
           Technical University of Denmark, DK-2800 Lyngby, Denmark, 1998. A
           copy of this report is included in the ``stogo`` subdirectory of
           NLopt as ``techreport.pdf``.


        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            randomized: If True, uses the randomized version 'GD_STOGO_RAND' instead of 'GD_STOGO'
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """

        if randomized is None:
            randomized = False

        if randomized:
            algorithm = nlopt.GD_STOGO_RAND
        else:
            algorithm = nlopt.GD_STOGO
        super().__init__(
            name=name,
            algorithm=algorithm,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options=None,
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptESCHV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt ESCH",
    ):
        """Global minimizer using an evolutionary algorithm.

        This is a modified Evolutionary Algorithm for global optimization,
        developed by Carlos Henrique da Silva Santos’s and described in the
        following paper and Ph.D thesis:

        -  C. H. da Silva Santos, M. S. Gonçalves, and H. E. Hernandez-Figueroa,
           “Designing Novel Photonic Devices by Bio-Inspired Computing,” *IEEE
           Photonics Technology Letters* **22** (15), pp. 1177–1179 (2010).

        .. raw:: html

           <!-- -->

        -  C. H. da Silva Santos, “`Parallel and Bio-Inspired Computing Applied
           to Analyze Microwave and Photonic Metamaterial Strucutures`_,”
           Ph.D. thesis, University of Campinas, (2010).

        The algorithm is adapted from ideas described in:

        -  H.-G. Beyer and H.-P. Schwefel, “Evolution Strategies: A
           Comprehensive Introduction,” *Journal Natural Computing*, **1** (1),
           pp. 3–52 (2002_.

        .. raw:: html

           <!-- -->

        -  Ingo Rechenberg, “Evolutionsstrategie – Optimierung technischer
           Systeme nach Prinzipien der biologischen Evolution,” Ph.D. thesis
           (1971), Reprinted by Fromman-Holzboog (1973).

        .. _Parallel and Bio-Inspired Computing Applied to Analyze Microwave and Photonic Metamaterial Strucutures: http://www.bibliotecadigital.unicamp.br/document/?code=000767537&opt=4&lg=en_US

        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
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
            maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
            strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
            criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
            name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """

        algorithm = nlopt.GN_ESCH

        super().__init__(
            name=name,
            algorithm=algorithm,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options=None,
            strategy=strategy,
            maxiter=maxiter,
        )


class NLoptISRESV1(NLoptBaseMinimizerV1):
    def __init__(
        self,
        tol: float | None = None,
        population: int | None = None,
        verbosity: int | None = None,
        maxiter: int | str | None = None,
        strategy: ZfitStrategy | None = None,
        criterion: ConvergenceCriterion | None = None,
        name: str = "NLopt ISRES",
    ):
        """Improved Stochastic Ranking Evolution Strategy using a mutation rule and differential variation.

         The evolution strategy is based on a combination of a mutation rule (with a log-normal step-size update and
         exponential smoothing) and differential variation (a Nelder–Mead-like update rule).
         The fitness ranking is simply via the objective function for problems without nonlinear constraints,
         but when nonlinear constraints are included the stochastic ranking proposed by Runarsson and Yao is employed.

         The NLopt implementation is based on the method described in:

         - Thomas Philip Runarsson and Xin Yao,
          `"Search biases in constrained evolutionary optimization <http://www3.hi.is/~tpr/papers/RuYa05.pdf>"`_,
          *IEEE Trans. on Systems, Man, and Cybernetics Part C: Applications and Reviews*,
           vol. 35 (no. 2), pp. 233-243 (2005).

         It is a refinement of an earlier method described in:

         - Thomas P. Runarsson and Xin Yao,
           `"Stochastic ranking for constrained evolutionary optimization
           <http://www3.hi.is/~tpr/software/sres/Tec311r.pdf>"`_, *IEEE Trans. Evolutionary Computation*,
           vol. 4 (no. 3), pp. 284-294 (2000).

         The actual implementation is independent provided by S. G. Johnson (2009) based on the papers above.
         Runarsson also has his own Matlab implemention available from `his web page <http://www3.hi.is/~tpr>`_.


        |@doc:minimizer.nlopt.info| More information on the algorithm can be found
        `here <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_.

        This implenemtation uses internally the
        `NLopt library <https://nlopt.readthedocs.io/en/latest/>`_.
        It is a
        free/open-source library for nonlinear optimization,
        providing a common interface for a number of
        different free optimization routines available online as well as
        original implementations of various other algorithms. |@docend:minimizer.nlopt.info|

         Args:
             tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
             population: |@doc:minimizer.nlopt.population| The population size for the evolutionary algorithm. |@docend:minimizer.nlopt.population| Defaults to 20×(n+1) in n dimensions.
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
             maxiter: |@doc:minimizer.maxiter| Approximate number of iterations.
                   This corresponds to roughly the maximum number of
                   evaluations of the ``value``, 'gradient`` or ``hessian``. |@docend:minimizer.maxiter|
             strategy: |@doc:minimizer.strategy| A class of type ``ZfitStrategy`` that takes no
                   input arguments in the init. Determines the behavior of the minimizer in
                   certain situations, most notably when encountering
                   NaNs. It can also implement a callback function. |@docend:minimizer.strategy|
             criterion: |@doc:minimizer.criterion| Criterion of the minimum. This is an
                   estimated measure for the distance to the
                   minimum and can include the relative
                   or absolute changes of the parameters,
                   function value, gradients and more.
                   If the value of the criterion is smaller
                   than ``loss.errordef * tol``, the algorithm
                   stopps and it is assumed that the minimum
                   has been found. |@docend:minimizer.criterion|
             name: |@doc:minimizer.name| Human-readable name of the minimizer. |@docend:minimizer.name|
        """

        algorithm = nlopt.GN_ISRES
        if population is not None:
            minimizer_options = {"population": population}
        else:
            minimizer_options = None

        super().__init__(
            name=name,
            algorithm=algorithm,
            tol=tol,
            gradient=NOT_SUPPORTED,
            hessian=NOT_SUPPORTED,
            criterion=criterion,
            verbosity=verbosity,
            minimizer_options=minimizer_options,
            strategy=strategy,
            maxiter=maxiter,
        )

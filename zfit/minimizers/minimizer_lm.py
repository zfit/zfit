#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import zfit.z.numpy as znp

from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter, assign_values
from ..util.cache import GraphCachable
from ..util.exception import MaximumIterationReached
from .baseminimizer import BaseMinimizer, minimize_supports
from .fitresult import Approximations, FitResult
from .strategy import ZfitStrategy
from .termination import ConvergenceCriterion


class OptimizeStop(Exception):
    pass


class LevenbergMarquardt(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "LM"

    def __init__(
        self,
        tol: float | None = None,
        mode: int | None = None,
        rho_min: float | None = None,
        rho_max: float | None = None,
        verbosity: int | None = None,
        options: Mapping[str, object] | None = None,
        maxiter: int | None = None,
        criterion: ConvergenceCriterion | None = None,
        strategy: ZfitStrategy | None = None,
        name: str | None = None,
    ):
        """Levenberg-Marquardt minimizer for general non-linear minimization by interpolating between Gauss-Newton and
        Gradient descent optimization.

        LM minimizes a function by iteratively solving a locally linearized
        version of the problem. Using the gradient (g) and the Hessian (H) of
        the loss function, the algorithm determines a step (h) that minimizes
        the loss function by solving :math:`Hh = g`. This works perfectly in one
        step for linear problems, however for non-linear problems it may be
        unstable far from the minimum. Thus a scalar damping parameter (L) is
        introduced and the Hessian is modified based on this damping. The form
        of the modification depends on the ``mode`` parameter, where the
        simplest example is :math:`H' = H + L \\mathbb{I}` (where
        :math:`\\mathbb{I}` is the identity). For a clarifying discussion of the
        LM algorithm see: Gavin, Henri 2016
        (https://msulaiman.org/onewebmedia/LM%20Method%20matlab%20codes%20and%20implementation.pdf)

        Args:
            tol: |@doc:minimizer.tol| Termination value for the
                   convergence/stopping criterion of the algorithm
                   in order to determine if the minimum has
                   been found. Defaults to 1e-3. |@docend:minimizer.tol|
            mode: The mode of the LM algorithm. The default mode (0) is the
                simplest form of the LM algorithm. Other modes may be implemented
                in the future.
            rho_min: The minimum acceptable value of the ratio of the actual
                reduction in the loss function to the expected reduction. If the
                ratio is less than this value, the damping parameter is increased.
            rho_max: The maximum acceptable value of the ratio of the actual
                reduction in the loss function to the expected reduction. If the
                ratio is greater than this value, the damping parameter is
                decreased.
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
            options: |@doc:minimizer.options||@docend:minimizer.options|
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
            name: |@doc:minimizer.name||@docend:minimizer.name

        """

        mode = 0 if mode is None else mode
        if mode != 0:
            msg = "Only mode 0 is currently implemented"
            raise NotImplementedError(msg)
        rho_min = 0.1 if rho_min is None else rho_min
        if rho_min <= 0:
            msg = "rho_min has to be > 0"
            raise ValueError(msg)
        rho_max = 2.0 if rho_max is None else rho_max
        if rho_max <= 1:
            msg = "rho_max has to be > 1"
            raise ValueError(msg)

        self.mode = mode
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.Lup = 11
        self.Ldn = 9
        super().__init__(
            name=name,
            strategy=strategy,
            tol=tol,
            verbosity=verbosity,
            criterion=criterion,
            maxiter=maxiter,
            minimizer_options=options,
        )

    @staticmethod
    def _damped_hess0(hess, L):
        I = znp.eye(hess.shape[0])  # noqa: E741
        D = znp.ones_like(hess) - I
        return hess * (I + D / (1 + L)) + L * I * (1 + znp.diag(hess))

    def _mode0_step(self, loss, params, L):
        init_chi2, grad = loss.value_gradient(params)
        hess = loss.hessian(params)
        grad = grad[..., None]  # right shape for the solve
        best = (znp.zeros_like(params), init_chi2, L)
        scary_best = (None, init_chi2, L)
        direction = "none"
        nostep = True
        for _ in range(10):
            # Determine damped hessian
            damped_hess = self._damped_hess0(hess, L)
            # Solve for step with minimum chi^2
            h = znp.linalg.solve(damped_hess, -grad)

            # Calculate expected improvement in chi^2
            expected_improvement = -znp.transpose(h) @ damped_hess @ h - 2 * znp.transpose(h) @ grad

            # Calculate new chi^2
            params1 = params + h[:, 0]
            chi2 = loss.value(params1)
            # Skip if chi^2 is nan
            if not np.isfinite(chi2):
                L *= self.Lup
                if direction == "better":
                    break
                direction = "worse"
                continue

            # Keep track of any chi^2 improvement
            if chi2 <= scary_best[1]:
                scary_best = (h, chi2, L)

            # Check if chi^2 improved the expected amount
            rho = (init_chi2 - chi2) / expected_improvement
            if rho < self.rho_min or rho > self.rho_max:
                L *= self.Lup
                if L > 1e9 or direction == "better":
                    break
                direction = "worse"
                continue

            # Check for new best chi^2
            if chi2 < best[1]:
                best = (h, chi2, L)
                L /= self.Ldn
                nostep = False
                if L < 1e-9 or direction == "worse":
                    break
                direction = "better"
            elif chi2 >= best[1] and direction in ["worse", "none"] and L < 1e9:
                L *= self.Lup
                direction = "worse"
                continue
            else:  # chi2 >= best[1] and direction == "better"
                break

            # Break if step already very successful (> 10% improvement)
            if (init_chi2 - best[1]) / init_chi2 > 0.1:
                break

        if nostep:
            # use any improvement if a safe improvement could not be found
            if scary_best[0] is not None:
                return scary_best
            msg = "No step found"
            raise OptimizeStop(msg)
        return {"step": best, "hessian": damped_hess, "gradient": grad[:, 0]}

    @minimize_supports(init=True)
    def _minimize(self, loss: ZfitLoss, params: list[Parameter], init):
        if init:
            assign_values(params=params, values=init)

        evaluator = self.create_evaluator(loss=loss, params=params)
        criterion = self.create_criterion(loss=loss, params=params)
        loss_history = [evaluator.value(params)]
        L = 1.0
        success = False
        paramvals = znp.asarray(params)
        step = None
        approx = None
        iterator = range(self.get_maxiter(n=len(params)))
        for _niter in iterator:
            try:
                new_point = self._mode0_step(evaluator, paramvals, L)
                step = new_point["step"]
            except OptimizeStop:
                if self.verbosity >= 7:
                    print(f"OptimizeStop, Iteration {_niter}, loss: {loss_history[-1]}")
                break
            L = step[2]
            loss_history.append(step[1])

            paramvals += step[0][:, 0]
            approx = Approximations(params=params, gradient=new_point.get("gradient"), hessian=new_point.get("hessian"))
            tempres = FitResult(
                loss=loss,
                params=dict(zip(params, paramvals)),
                minimizer=self,
                valid=False,
                criterion=criterion,
                edm=-999,
                fminopt=loss_history[-1],
                approx=approx,
            )
            success = criterion.converged(tempres)
            if self.verbosity >= 7:
                print(
                    f"Iteration {_niter}, loss: {loss_history[-1]}, success: {success}, criterion: {criterion.last_value}"
                )
            if success:
                if self.verbosity >= 6:
                    print(f"Converged with criterion: {criterion.last_value} < {self.tol}")
                break
        else:
            msg = f"Maximum number of iterations ({self.maxiter}) reached."
            raise MaximumIterationReached(msg)

        msg = "No step taken. This should not happen?"
        assert step is not None, msg
        assign_values(params, paramvals)

        return FitResult(
            loss=loss,
            params={p: float(p.value()) for p in params},
            edm=criterion.last_value,
            fminopt=loss_history[-1],
            minimizer=self,
            valid=success,
            criterion=criterion,
            approx=approx,
            evaluator=evaluator,
        )

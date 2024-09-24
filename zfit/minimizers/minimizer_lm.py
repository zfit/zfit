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
        mode: int = 0,
        rho_min: float = 0.1,
        rho_max: float = 2.0,
        verbosity: int | None = None,
        options: Mapping[str, object] | None = None,
        maxiter: int = 1000,
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
        """

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
        iterator = range(self.maxiter)
        for _niter in iterator:
            try:
                new_point = self._mode0_step(evaluator, paramvals, L)
                step = new_point["step"]
            except OptimizeStop:
                if self.verbosity >= 7:
                    pass
                break
            L = step[2]
            loss_history.append(step[1])

            paramvals = step[0][:, 0]
            Approximations(params=params, gradient=new_point.get("gradient"), hessian=new_point.get("hessian"))
            tempres = FitResult(
                loss=loss,
                params=dict(zip(params, paramvals)),
                minimizer=self,
                valid=False,
                criterion=criterion,
                edm=-999,
                fminopt=loss_history[-1],
                # approx=approx,
            )
            # success = (len(loss_history) < 3 or (loss_history[-3] - loss_history[-1]) / loss_history[
            #     -1] < self.tol) and criterion.converged(tempres)
            success = criterion.converged(tempres)
            if self.verbosity >= 7:
                print(
                    f"Iteration {_niter}, loss: {loss_history[-1]}, success: {success}, criterion: {criterion.last_value}"
                )
            if success:
                if self.verbosity >= 6:
                    pass
                break
        else:
            msg = f"Maximum number of iterations ({self.maxiter}) reached."
            raise MaximumIterationReached(msg)

        msg = "No step taken. This should not happen?"
        assert step is not None, msg
        assign_values(params, params + step[0][:, 0])

        return FitResult(
            loss=loss,
            params={p: p.value() for p in params},
            edm=criterion.last_value,
            fminopt=loss_history[-1],
            minimizer=self,
            valid=success,
            criterion=criterion,
        )

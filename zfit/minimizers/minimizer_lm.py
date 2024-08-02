from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import zfit.z.numpy as znp

from ..core.interfaces import ZfitLoss
from ..core.parameter import Parameter, assign_values
from ..util.cache import GraphCachable
from .baseminimizer import BaseMinimizer, minimize_supports
from .fitresult import FitResult
from .strategy import ZfitStrategy
from .termination import ConvergenceCriterion
from ..util.exception import MaximumIterationReached


class OptimizeStop(Exception):
    pass


class LevenbergMarquardt(BaseMinimizer, GraphCachable):
    _DEFAULT_name = "LM"

    def __init__(
        self,
        tol: float | None = None,
        mode: int = 0,
        confidence: float = 2.0,
        verbosity: int | None = None,
        options: Mapping[str, object] | None = None,
        maxiter: int = 100,
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
        self.confidence = confidence
        self.Lup = 10  # TODO: just added this adhoc
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
        I = znp.eye(hess.shape[0])
        D = znp.ones_like(hess) - I
        return hess * (I + D / (1 + L)) + L * I * (1 + znp.diag(hess))

    def _mode0_step(self, loss, params, L):
        init_chi2, grad = loss.value_gradient(params)
        hess = loss.hessian(params)
        grad = - grad[..., None]  # right shape for the solve
        best = (znp.zeros_like(params), init_chi2, L)
        scarry_best = (None, init_chi2, L)
        direction = "none"
        nostep = True
        for _ in range(10):
            # Determine damped hessian
            damped_hess = self._damped_hess0(hess, L)

            # Solve for step with minimum chi^2
            h = znp.linalg.solve(damped_hess, grad)

            # Calculate expected improvement in chi^2
            expected_improvement = znp.transpose(h) @ damped_hess @ h - 2 * znp.transpose(h) @ grad

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
            if chi2 <= scarry_best[1]:
                scarry_best = (h, chi2, L)

            # Check if chi^2 improved too much
            if ((init_chi2 - chi2) / init_chi2) > (self.confidence * expected_improvement):
                L = L * self.Lup
                if direction == "better":
                    break
                direction = "worse"
                continue

            # Check for new best chi^2
            if chi2 < best[1]:
                best = (h, chi2, L)
                L = L / self.Ldn
                nostep = False
                if L < 1e-8 or direction == "worse":
                    break
                direction = "better"
            elif chi2 >= best[1] and direction in ["worse", "none"]:
                L = L * self.Lup
                direction = "worse"
                continue
            else:  # chi2 >= best[1] and direction == "better"
                break

            # Break if step already very successful (> 10% improvement)
            if (init_chi2 - best[1]) / init_chi2 > 0.1:
                break

        if nostep:
            # use any improvement if a safe improvement could not be found
            if scarry_best[0] is not None:
                return scarry_best
            msg = "No step found"
            raise OptimizeStop(msg)
        return best

    @minimize_supports(init=True)
    def _minimize(self, loss: ZfitLoss, params: list[Parameter], init):
        if init:
            assign_values(params=params, values=init)

        evaluator = self.create_evaluator(loss=loss, params=params)

        loss_history = [evaluator.value(params)]
        L = znp.eye(len(params))
        success = False
        step = None
        for _ in range(self.maxiter):
            try:
                step = self._mode0_step(evaluator, params, L)
            except OptimizeStop:
                break

            L = step[2]
            params = params + step[0][:, 0]
            loss_history.append(step[1])
            if len(loss_history) >= 3:
                # Loss no longer updating and L is small, minimum reached
                if (loss_history[-3] - loss_history[-1]) / loss_history[-1] < self.tol and znp.all(znp.diag(L) < 0.1):
                    success = True
                    break
        else:
            raise IOError(f"Maximum number of iterations ({self.maxiter}) reached.")

        if step is None:
            msg = "No step taken. This should not happen?"
            raise IOError(msg)
        return FitResult(loss=step[1], params=params, minimizer=self, valid=success)

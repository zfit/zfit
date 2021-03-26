#  Copyright (c) 2021 zfit
import abc
from typing import Optional

import numpy as np

from ..core.interfaces import ZfitLoss
from ..util import ztyping
from ..util.checks import Singleton


class ConvergenceCriterion(abc.ABC):

    def __init__(self, tol: float,
                 loss: ZfitLoss,
                 params: ztyping.ParamTypeInput,
                 name: str):
        """A generic convergence criterion to be subclassed.

        Args:
            tol: tolerance to stop the minimization. This value is multiplied by the errordef of the loss.
            loss:
            params ():
            name ():
        """
        super().__init__()
        if not isinstance(loss, ZfitLoss):
            raise TypeError("loss has to be ZfitLoss")
        self.loss = loss
        self.tol = tol * loss.errordef
        self.params = params
        self.name = name
        self.last_value = CRITERION_NOT_AVAILABLE

    def converged(self, result: "zfit.core.fitresult.FitResult") -> bool:
        """Calculate the criterion and check if it is below the tolerance.

        Args:
            result: Return the result which contains all the information

        Returns:
        """
        value = self.calculate(result)
        return value < self.tol

    def calculate(self, result: "zfit.core.fitresult.FitResult"):
        """Evaluate the convergence criterion and store it in `last_value`

        Args:
            result ():
        """
        value = self._calculate(result=result)
        self.last_value = value
        return value

    @abc.abstractmethod
    def _calculate(self, result: "zfit.core.fitresult.FitResult") -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<ConvergenceCriterion {self.name}>"


# @numba.njit(numba.float64(numba.float64[:], numba.float64[:, :]))
def calculate_edm(grad, inv_hesse):
    # grad = np.array(grad)
    return grad @ inv_hesse @ grad / 2


class EDM(ConvergenceCriterion):

    def __init__(self,
                 tol: float,
                 loss: ZfitLoss,
                 params: ztyping.ParamTypeInput,
                 name: Optional[str] = "edm"):

        super().__init__(tol=tol, loss=loss, params=params,
                         name=name)

    def _calculate(self, result) -> float:
        loss = result.loss
        params = list(result.params)
        grad = result.approx.gradient()
        if grad is None:
            grad = loss.gradient(params)
        grad = np.array(grad)
        inv_hessian = result.approx.inv_hessian(invert=True)
        if inv_hessian is None:
            hessian = loss.hessian(params)
            inv_hessian = np.linalg.inv(hessian)
        return calculate_edm(grad, inv_hessian)


class CriterionNotAvailable(Singleton):

    def __bool__(self):
        return False

    def __eq__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __add__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __pow__(self, power, modulo=None):
        return self

    def __repr__(self):
        return "<EDM_not_available>"


CRITERION_NOT_AVAILABLE = CriterionNotAvailable()

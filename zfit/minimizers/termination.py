#  Copyright (c) 2021 zfit
import abc
from typing import Optional

import numpy as np

from ..core.interfaces import ZfitLoss
from ..util import ztyping


class ConvergenceCriterion(abc.ABC):

    def __init__(self, tolerance: float, loss: ZfitLoss, params: ztyping.ParamTypeInput, name: str):
        super().__init__()
        if not isinstance(loss, ZfitLoss):
            raise TypeError("loss has to be ZfitLoss")
        self.loss = loss
        self.tolerance = tolerance
        self.params = params
        self.name = name

    def convergedV1(self, value, xvalues, grad=None, hesse=None, inv_hesse=None, **kwargs) -> bool:
        self.last_value = self.calculateV1(value, xvalues=xvalues, grad=grad, hesse=hesse, inv_hesse=inv_hesse,
                                           **kwargs)
        return self.last_value < self.tolerance

    @abc.abstractmethod
    def calculateV1(self, value, xvalues, grad, hesse, inv_hesse, **kwargs):
        raise NotImplementedError


def calculate_edm(grad, inv_hesse):
    return grad @ inv_hesse @ grad / 2


class EDM(ConvergenceCriterion):

    def __init__(self, tolerance: float, loss: ZfitLoss, params: ztyping.ParamTypeInput, name: Optional[str] = None):
        if name is None:
            name = "edm"
        super().__init__(tolerance, loss, params, name)

    def calculateV1(self, value, xvalues, grad, hesse=None, inv_hesse=None, **kwargs) -> float:
        del value
        if callable(grad):
            grad = grad()
        if callable(hesse):
            hesse = hesse()
        if callable(inv_hesse):
            inv_hesse = inv_hesse()
        if inv_hesse is None:
            if hesse is None:
                raise RuntimeError("Need hesse or inv_hesse for convergence criterion:")
            else:
                inv_hesse = np.linalg.inv(hesse)
        edm = calculate_edm(grad, inv_hesse)
        return edm

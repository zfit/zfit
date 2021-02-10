#  Copyright (c) 2021 zfit
import abc
from typing import Optional, Tuple

import numpy as np

from .fitresult import FitResult


class ConvergenceCriterion(abc.ABC):

    def __init__(self, tolerance: float, name: str):
        super().__init__()
        self.tolerance = tolerance
        self.name = name

    def converged(self, result: FitResult) -> Tuple[bool, float]:
        """Calculate the criterion and check if it is below the tolerance, return both.

        Args:
            result: Return the result which contains all the information

        Returns:

        """
        value = self.calculate(result)
        return value < self.tolerance, value

    @abc.abstractmethod
    def calculate(self, result: FitResult):
        raise NotImplementedError


def calculate_edm(grad, inv_hesse):
    return grad @ inv_hesse @ grad / 2


class EDM(ConvergenceCriterion):

    def __init__(self, tolerance: float, name: Optional[str] = None):
        if name is None:
            name = "edm"
        super().__init__(tolerance=tolerance, name=name)

    def calculate(self, result) -> float:

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

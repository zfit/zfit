#  Copyright (c) 2023 zfit

from __future__ import annotations

import abc
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from .fitresult import FitResult
from ..core.interfaces import ZfitLoss, ZfitParameter
from ..settings import run
from ..util import ztyping


class FailMinimizeNaN(Exception):
    pass


class ZfitStrategy(abc.ABC):
    @abstractmethod
    def minimize_nan(
        self, loss: ZfitLoss, params: ztyping.ParamTypeInput, values: Mapping = None
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def callback(
        self,
        value: float | None,
        gradient: np.ndarray | None,
        hessian: np.ndarray | None,
        params: list[ZfitParameter],
        loss: ZfitLoss,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseStrategy(ZfitStrategy):
    def __init__(self) -> None:
        self.fit_result = None
        self.error = None
        super().__init__()

    def minimize_nan(
        self, loss: ZfitLoss, params: ztyping.ParamTypeInput, values: Mapping = None
    ) -> float:
        print(
            "The minimization failed due to too many NaNs being produced in the loss."
            "This is most probably caused by negative"
            " values returned from the PDF. Changing the initial values/stepsize of the parameters can solve this"
            " problem. Also check your model (if custom) for problems. For more information,"
            " visit https://github.com/zfit/zfit/wiki/FAQ#fitting-and-minimization"
        )
        raise FailMinimizeNaN()

    def callback(self, value, gradient, hessian, params, loss):
        del params
        return value, gradient, hessian

    def __str__(self) -> str:
        return repr(self.__class__)[:-2].split(".")[-1]


class ToyStrategyFail(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.fit_result = FitResult(
            params={},
            edm=None,
            fmin=None,
            status=None,
            converged=False,
            info={},
            valid=False,
            message="NaN produced, ToyStrategy fails",
            niter=None,
            loss=None,
            minimizer=None,
            criterion=None,
        )

    def minimize_nan(
        self, loss: ZfitLoss, params: ztyping.ParamTypeInput, values: Mapping = None
    ) -> float:
        param_vals = run(params)
        param_vals = OrderedDict(
            (param, value) for param, value in zip(params, param_vals)
        )
        self.fit_result = FitResult(
            params=param_vals,
            edm=None,
            fmin=None,
            status=9,
            converged=False,
            info={},
            loss=loss,
            valid=False,
            message="Failed on too manf NaNs",
            niter=None,
            criterion=None,
            minimizer=None,
        )
        raise FailMinimizeNaN()


def make_pushback_strategy(
    nan_penalty: float | int = 100,
    nan_tol: int = 30,
    base: object | ZfitStrategy = BaseStrategy,
):
    class PushbackStrategy(base):
        def __init__(self):
            """Pushback by adding `nan_penalty * counter` to the loss if NaNs are encountered.

            The counter indicates how many NaNs occurred in a row. The `nan_tol` is the upper limit, if this is
            exceeded, the fallback will be used and an error is raised.

            Args:
                nan_penalty: Value to add to the previous loss in order to penalize the step taken.
                nan_tol: If the number of NaNs encountered in a row exceeds this number, the fallback is used.
            """
            super().__init__()
            self.nan_penalty = nan_penalty
            self.nan_tol = nan_tol

        def minimize_nan(
            self, loss: ZfitLoss, params: ztyping.ParamTypeInput, values: Mapping = None
        ) -> float:
            assert (
                "nan_counter" in values
            ), "'nan_counter' not in values, minimizer not correctly implemented"
            nan_counter = values["nan_counter"]
            if nan_counter < self.nan_tol:
                last_loss = values.get("old_loss")
                last_grad = values.get("old_grad")
                if last_grad is not None:
                    last_grad = -last_grad
                if last_loss is not None:
                    loss_evaluated = last_loss + self.nan_penalty * nan_counter
                else:
                    loss_evaluated = values.get("loss")
                if isinstance(loss_evaluated, str):
                    raise RuntimeError("Loss starts already with NaN, cannot minimize.")
                return loss_evaluated, last_grad
            else:
                super().minimize_nan(loss=loss, params=params, values=values)

    return PushbackStrategy


PushbackStrategy = make_pushback_strategy()


class DefaultToyStrategy(PushbackStrategy, ToyStrategyFail):
    """Same as :py:class:`PushbackStrategy`, but does not raise an error on full failure, instead return an invalid
    FitResult.

    This can be useful for toy studies, where multiple fits are done and a failure should simply be counted as a failure
    instead of rising an error.
    """

    pass

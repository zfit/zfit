#  Copyright (c) 2024 zfit
from __future__ import annotations

from .. import z
from ..core.interfaces import ZfitData, ZfitFunc, ZfitSpace
from ..core.space import convert_to_space


class FuncLoss:
    def __init__(self, func: ZfitFunc, data: ZfitData, *, other=None):
        self.func = func
        self.data = data
        self.target = func.output
        self.other = other

    def __call__(self, params=None):
        return self.value(params=params)

    def value(self, *, params=None):
        """Compute the value of the loss function."""
        # todo: preprocess params
        return self._call_value(params=params)

    # @z.function(wraps="funcloss")
    def _call_value(self, params):
        func = self.func
        x = self.data.with_obs(func.space)
        y = self.data[self.target]
        pred = func(x, params=params)[self.target]
        args = [pred, y]
        if (other := self.other) is not None:
            other = self.data[other]
            args.append(other)

        return self._loss(*args)

    def _loss(self, pred, target, other=None):
        raise NotImplementedError

    def get_params(self, *, floating=True, extract_independent=True):
        return self.func.get_params(floating=floating, extract_independent=extract_independent)


class Chi2(FuncLoss):
    def __init__(self, func: ZfitFunc, data: ZfitData, uncertainty: str | ZfitSpace | None = None):
        if uncertainty is not None:
            uncertainty = convert_to_space(uncertainty)
            if uncertainty.n_obs > 1:
                msg = "Chi2 can only be used with a 1D uncertainty."
                raise ValueError(msg)
        super().__init__(func=func, data=data, other=uncertainty)

    def _loss(self, pred, target, other=None):
        diff = pred - target
        if other is not None:
            diff /= other
        return z.reduce_sum(z.square(diff))

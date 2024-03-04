#  Copyright (c) 2023 zfit
"""Baseclass for ``Function``. Inherits from Model.

TODO(Mayou36): subclassing?
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

import abc
import typing

from ..util.exception import ShapeIncompatibleError, SpecificFunctionNotImplemented
from .basemodel import BaseModel
from .interfaces import ZfitFunc
from ..settings import ztypes
from ..util import ztyping


class BaseFuncV1(BaseModel, ZfitFunc):
    def __init__(
        self,
        obs=None,
        dtype: type = ztypes.float,
        name: str = "BaseFunc",
        params: typing.Any = None,
    ):
        """TODO(docs): explain subclassing."""
        super().__init__(obs=obs, dtype=dtype, name=name, params=params)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.func(x=x)

    def _func_to_sample_from(self, x):
        return self.func(x=x)

    # TODO(Mayou36): how to deal with copy properly?
    def copy(self, **override_params):
        new_params = self.params
        new_params.update(override_params)
        return type(self)(new_params)

    def gradient(
        self,
        x: ztyping.XType,
        norm: ztyping.LimitsType = None,
        params: ztyping.ParamsTypeOpt = None,
    ):
        # TODO(Mayou36): well, really needed... this gradient?
        raise NotImplementedError("What do you need? Use tf.gradient...")

    @abc.abstractmethod
    def _func(self, x):
        raise SpecificFunctionNotImplemented

    def func(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        """The function evaluated at ``x``.

        Args:
            x:
            name:

        Returns:
             # TODO(Mayou36): or dataset? Update: rather not, what would obs be?
        """
        with self._convert_sort_x(x) as x:
            return self._single_hook_value(x=x, name=name)

    def _single_hook_value(self, x, name):
        return self._hook_value(x, name)

    def _hook_value(self, x, name="_hook_value"):
        return self._call_value(x=x, name=name)

    def _call_value(self, x, name):
        try:
            return self._func(x=x)
        except ValueError as error:
            raise ShapeIncompatibleError(
                "Most probably, the number of obs the func was designed for"
                "does not coincide with the `n_obs` from the `space`/`obs`"
                "it received on initialization."
            ) from error

    def as_pdf(self) -> zfit.core.interfaces.ZfitPDF:
        """Create a PDF out of the function.

        Returns:
            A PDF with the current function as the unnormalized probability.
        """
        from zfit.core.operations import convert_func_to_pdf

        return convert_func_to_pdf(func=self)

    def _check_input_norm_range_default(
        self, norm_range, caller_name="", none_is_error=True
    ):  # TODO(Mayou36): default
        return self._check_input_norm(norm=norm_range, none_is_error=none_is_error)

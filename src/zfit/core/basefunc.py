#  Copyright (c) 2025 zfit
"""Baseclass for ``Function``. Inherits from Model.

TODO(Mayou36): subclassing?
"""

from __future__ import annotations

import abc
import typing
from typing import Any

from zfit._interfaces import ZfitFunc

from ..settings import ztypes
from ..util import ztyping
from ..util.exception import ShapeIncompatibleError, SpecificFunctionNotImplemented
from .basemodel import BaseModel

if typing.TYPE_CHECKING:
    import zfit


class BaseFuncV1(BaseModel, ZfitFunc):
    def __init__(
        self,
        obs=None,
        dtype: type = ztypes.float,
        name: str = "BaseFunc",
        params: Any = None,
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

    @abc.abstractmethod
    def _func(self, x):
        raise SpecificFunctionNotImplemented

    def func(
        self,
        x: ztyping.XType,
        name: str = "value",
        *,
        params: ztyping.ParamsTypeInput | None = None,
    ) -> ztyping.XType:
        """The function evaluated at ``x``.

        Args:
            x:
            name:

        Returns:
             # TODO(Mayou36): or dataset? Update: rather not, what would obs be?
        """
        with self._convert_sort_x(x) as xclean, self._check_set_input_params(params=params):
            return self._single_hook_value(x=xclean, name=name)

    def _single_hook_value(self, x, name):
        return self._hook_value(x, name)

    def _hook_value(self, x, name="_hook_value"):
        return self._call_value(x=x, name=name)

    def _call_value(self, x, name):  # noqa: ARG002
        try:
            return self._func(x=x)
        except ValueError as error:
            msg = (
                "Most probably, the number of obs the func was designed for"
                "does not coincide with the `n_obs` from the `space`/`obs`"
                "it received on initialization."
            )
            raise ShapeIncompatibleError(msg) from error

    def as_pdf(self) -> zfit.interfaces.ZfitPDF:
        """Create a PDF out of the function.

        Returns:
            A PDF with the current function as the unnormalized probability.
        """
        from zfit.core.operations import convert_func_to_pdf  # noqa: PLC0415

        return convert_func_to_pdf(func=self)

    def _check_input_norm_range_default(self, norm, caller_name="", none_is_error=True):  # TODO(Mayou36): default
        del caller_name  # unused
        return self._check_input_norm(norm=norm, none_is_error=none_is_error)

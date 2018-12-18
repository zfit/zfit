"""Baseclass for `Function`. Inherits from Model.

TODO(Mayou36): subclassing?
"""
import abc
import typing

import tensorflow as tf

from .basemodel import BaseModel
from .interfaces import ZfitFunc
from ..settings import types as ztypes
from ..util import ztyping


class BaseFunc(BaseModel, ZfitFunc):

    def __init__(self, obs=None, dtype: typing.Type = ztypes.float, name: str = "BaseFunc",
                 parameters: typing.Any = None):
        """TODO(docs): explain subclassing"""
        super().__init__(obs=obs, dtype=dtype, name=name, parameters=parameters)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.value(x=x)

    def _func_to_sample_from(self, x):
        return self.value(x=x)

    # TODO(Mayou36): how to deal with copy properly?
    def copy(self, **override_parameters):
        new_params = self.parameters
        new_params.update(override_parameters)
        return type(self)(new_params)

    def gradient(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, params: ztyping.ParamsType = None):
        # TODO(Mayou36): well, really needed... this gradient?
        raise NotImplementedError("What do you need? Use tf.gradient...")

    @abc.abstractmethod
    def _value(self, x):
        raise NotImplementedError

    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:  # TODO(naming): rename to `func`?
        """The function evaluated at `x`.

        Args:
            x (`Data`):
            name (str):

        Returns:
            tf.Tensor:  # TODO(Mayou36): or dataset?
        """
        with self._convert_sort_x(x):
            return self._single_hook_value(x=x, name=name)

    def _single_hook_value(self, x, name):
        return self._hook_value(x, name)

    def _hook_value(self, x, name='_hook_value'):
        return self._call_value(x=x, name=name)

    def _call_value(self, x, name):
        with self._name_scope(name, values=[x]):
            return self._value(x=x)

    def as_pdf(self):
        from zfit.core.operations import convert_func_to_pdf
        return convert_func_to_pdf(func=self)

    def _check_input_norm_range_default(self, norm_range, caller_name="", none_is_error=True):  # TODO(Mayou36): default
        # norm_range for functions?

        # if norm_range is None:
        #     norm_range = self.norm_range
        return self._check_input_norm_range(norm_range=norm_range, caller_name=caller_name, none_is_error=none_is_error)

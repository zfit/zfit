import abc
import typing

import tensorflow as tf

from .basemodel import BaseModel
from .interfaces import ZfitFunc
from ..util import ztyping
from ..settings import types as ztypes
from zfit import ztf


class BaseFunc(BaseModel, ZfitFunc):

    def __init__(self, dims=None, dtype: typing.Type = ztypes.float, name: str = "BaseFunc",
                 parameters: typing.Any = None):
        super().__init__(dims=dims, dtype=dtype, name=name, parameters=parameters)

    def _func_to_sample_from(self, x):
        return self.value(x=x)

    def _func_to_integrate(self, x: ztyping.XType):
        self._hook_value(x=x)

    def copy(self, **override_parameters):
        new_params = self.parameters
        new_params.update(override_parameters)
        return type(self)(new_params)

    def gradient(self, x: ztyping.XType, params: ztyping.ParamsType = None):
        raise NotImplementedError("What do you need? Use tf.gradient...")

    @abc.abstractmethod
    def _value(self, x):
        raise NotImplementedError

    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        return self._hook_value(x, name)

    def _hook_value(self, x, name='_hook_value'):
        return self._call_value(x=x, name=name)

    def _call_value(self, x, name):
        with self._name_scope(name, values=[x]):
            x = ztf.convert_to_tensor(x, name="x")
            return self._value(x=x)

    def as_pdf(self):
        from zfit.core.operations import convert_func_to_pdf
        return convert_func_to_pdf(func=self)

    def _check_input_norm_range_default(self, norm_range, caller_name="", none_is_error=True):
        if norm_range is None:
            norm_range = self.norm_range
        return self._check_input_norm_range(norm_range=norm_range, caller_name=caller_name, none_is_error=none_is_error)

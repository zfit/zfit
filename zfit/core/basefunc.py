import abc
import contextlib
from contextlib import suppress
import typing

import pep487
import tensorflow as tf

from .basemodel import BaseModel, _BaseModel_register_check_support, ZfitModel
from zfit.core.limits import Range, convert_to_range, no_norm_range, no_multiple_limits, supports
from zfit.util import ztyping
from ..settings import types as ztypes
from .parameter import FitParameter
from ..util import exception as zexception
from zfit import ztf


class ZfitFunc(ZfitModel):
    @abc.abstractmethod
    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        raise NotImplementedError


class BaseFunc(BaseModel, ZfitFunc):

    def __init__(self, dtype: typing.Type = ztypes.float, name: str = "BaseFunc",
                 reparameterization_type: bool = False,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True, graph_parents: tf.Graph = None, **parameters: typing.Any):
        super().__init__(dtype=dtype, name=name, reparameterization_type=reparameterization_type,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats, graph_parents=graph_parents, **parameters)

        self.n_dims = None

    def _func_to_sample_from(self, x):
        self._hook_value(x=x)

    def get_parameters(self):
        return self.parameters

    def _func_to_integrate(self, x: ztyping.XType):
        self._hook_value(x=x)

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

import itertools
from typing import Dict, Union, Callable, Iterable

import tensorflow as tf

from zfit.core.basefunc import BaseFunc
from zfit.core.basemodel import SimpleModelSubclassMixin
from zfit.util import ztyping

from ..core.basefunc import BaseFunc
from ..core.interfaces import ZfitModel, ZfitFunc
from ..models.basefunctor import FunctorMixin
from ..util import ztyping
from ..util.container import convert_to_container


class SimpleFunc(BaseFunc):

    def __init__(self, func: Callable, obs: ztyping.ObsTypeInput, name: str = "Function", **params):
        """Create a simple function out of of `func` with the observables `obs` depending on `parameters`.

        Args:
            func (function):
            obs (Union[str, Tuple[str]]):
            name (str):
            **params (): The parameters as keyword arguments. E.g. `mu=Parameter(...)`
        """
        super().__init__(name=name, obs=obs, params=params)
        self._value_func = self._check_input_x_function(func)

    def _func(self, x):
        return self._value_func(self, x)


class BaseFunctorFunc(FunctorMixin, BaseFunc):
    def __init__(self, funcs, name="BaseFunctorFunc", **kwargs):
        funcs = convert_to_container(funcs)
        params = {}
        for func in funcs:
            params.update(func.params)

        self.funcs = funcs
        super().__init__(name=name, models=self.funcs, params=params, **kwargs)

    def _get_dependents(self):  # TODO: change recursive to `only_floating`?
        dependents = super()._get_dependents()  # get the own parameter dependents
        func_dependents = self._extract_dependents(self.funcs)  # flatten
        return dependents.union(func_dependents)

    @property
    def _models(self) -> Dict[Union[float, int, str], ZfitModel]:
        return self.funcs


class SumFunc(BaseFunctorFunc):
    def __init__(self, funcs: Iterable[ZfitFunc], obs: ztyping.ObsTypeInput = None, name: str = "SumFunc", **kwargs):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _func(self, x):
        # sum_funcs = tf.add_n([func.value(x) for func in self.funcs])
        funcs = [func.func(x) for func in self.funcs]
        sum_funcs = tf.accumulate_n(funcs)
        return sum_funcs


class ProdFunc(BaseFunctorFunc):
    def __init__(self, funcs: Iterable[ZfitFunc], obs: ztyping.ObsTypeInput = None, name: str = "SumFunc", **kwargs):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _func(self, x):
        value = self.funcs[0].func(x)
        for func in self.funcs[1:]:
            value *= func.func(x)
        return value


class ZFunc(SimpleModelSubclassMixin, BaseFunc):
    def __init__(self, obs: ztyping.ObsTypeInput, name: str = "ZFunc", **params):
        super().__init__(obs=obs, name=name, **params)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._check_simple_model_subclass()

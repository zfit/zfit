#  Copyright (c) 2022 zfit

from __future__ import annotations

from collections.abc import Iterable, Callable

import tensorflow as tf

import zfit.z.numpy as znp
from ..core.basefunc import BaseFuncV1
from ..core.basemodel import SimpleModelSubclassMixin
from ..core.dependents import _extract_dependencies
from ..core.interfaces import ZfitFunc
from ..core.space import supports
from ..models.basefunctor import FunctorMixin
from ..util import ztyping
from ..util.container import convert_to_container


class SimpleFuncV1(BaseFuncV1):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        func: Callable,
        name: str = "Function",
        **params,
    ):
        """Create a simple function out of of `func` with the observables `obs` depending on `parameters`.

        Args:
            func:
            obs:
            name:
            **params: The parameters as keyword arguments. E.g. `mu=Parameter(...)`
        """
        super().__init__(name=name, obs=obs, params=params)
        self._value_func = self._check_input_x_function(func)

    def _func(self, x):
        try:
            return self._value_func(x)
        except TypeError:  # self requested, TODO maybe check signature?
            return self._value_func(self, x)


class BaseFunctorFuncV1(FunctorMixin, BaseFuncV1):
    def __init__(self, funcs, name="BaseFunctorFunc", params=None, **kwargs):
        funcs = convert_to_container(funcs)
        if params is None:
            params = {}
        # for func in funcs:
        #     params.update(func.params)

        self.funcs = funcs
        super().__init__(name=name, models=self.funcs, params=params, **kwargs)
        self._models = self.funcs

    def _get_dependencies(self):  # TODO: change recursive to `only_floating`?
        dependents = super()._get_dependencies()  # get the own parameter dependents
        func_dependents = _extract_dependencies(self.funcs)  # flatten
        return dependents.union(func_dependents)


class SumFunc(BaseFunctorFuncV1):
    def __init__(
        self,
        funcs: Iterable[ZfitFunc],
        obs: ztyping.ObsTypeInput = None,
        name: str = "SumFunc",
        **kwargs,
    ):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _func(self, x):
        # sum_funcs = tf.add_n([func.value(x) for func in self.funcs])
        funcs = [func.func(x) for func in self.funcs]
        sum_funcs = tf.math.accumulate_n(funcs)
        return sum_funcs

    @supports()
    def _analytic_integrate(self, limits, norm):
        # below may raises AnalyticIntegralNotImplementedError, that's fine. We don't wanna catch that.
        integrals = [
            func.analytic_integrate(limits=limits, norm=norm) for func in self.funcs
        ]
        return tf.math.accumulate_n(integrals)


class ProdFunc(BaseFunctorFuncV1):
    def __init__(
        self,
        funcs: Iterable[ZfitFunc],
        obs: ztyping.ObsTypeInput = None,
        name: str = "SumFunc",
        **kwargs,
    ):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _func(self, x):
        funcs = [func.func(x) for func in self.funcs]
        product = znp.prod(funcs, axis=0)
        return product


class ZFuncV1(SimpleModelSubclassMixin, BaseFuncV1):
    def __init__(self, obs: ztyping.ObsTypeInput, name: str = "ZFunc", **params):
        super().__init__(obs=obs, name=name, **params)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._check_simple_model_subclass()

import itertools
from typing import Dict, Union

import tensorflow as tf

from zfit.core.basefunc import BaseFunc
from zfit.core.interfaces import ZfitModel
from zfit.models.basefunctor import FunctorMixin
from zfit.util.container import convert_to_container


class SimpleFunc(BaseFunc):

    def __init__(self, func, obs, name="Function", n_dims=None, **parameters):
        super().__init__(name=name, obs=obs, parameters=parameters)
        self._value_func = self._check_input_x_function(func)
        self._user_n_dims = n_dims

    def _value(self, x):
        return self._value_func(x)

    @property
    def _n_dims(self):
        n_dims = self._user_n_dims
        if n_dims is None:
            if self.axes is not None:
                n_dims = len(self.axes)
        return n_dims


class BaseFunctorFunc(FunctorMixin, BaseFunc):
    def __init__(self, funcs, name="BaseFunctorFunc", **kwargs):
        funcs = convert_to_container(funcs)
        params = {}
        for func in funcs:
            params.update(func.parameters)

        self.funcs = funcs
        super().__init__(name=name, models=self.funcs, parameters=params, **kwargs)

    def _get_dependents(self):  # TODO: change recursive to `only_floating`?
        dependents = super()._get_dependents()  # get the own parameter dependents
        func_dependents = self._extract_dependents(self.funcs)  # flatten
        return dependents.union(func_dependents)

    @property
    def _models(self) -> Dict[Union[float, int, str], ZfitModel]:
        return self.funcs


class SumFunc(BaseFunctorFunc):
    def __init__(self, funcs, obs=None, name="SumFunc", **kwargs):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _value(self, x):
        # sum_funcs = tf.add_n([func.value(x) for func in self.funcs])
        funcs = [func.value(x) for func in self.funcs]
        sum_funcs = tf.accumulate_n(funcs)
        return sum_funcs

    @property
    def _n_dims(self):
        return self._space.n_obs


class ProdFunc(BaseFunctorFunc):
    def __init__(self, funcs, obs=None, name="SumFunc", **kwargs):
        super().__init__(funcs=funcs, obs=obs, name=name, **kwargs)

    def _value(self, x):
        value = self.funcs[0].value(x)
        for func in self.funcs[1:]:
            value *= func.value(x)
        return value

    @property
    def _n_dims(self):
        return self._space.n_obs  # TODO(mayou36): working? implement like this is superclass?

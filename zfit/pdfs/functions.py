import itertools

import tensorflow as tf

from zfit.core.basefunc import BaseFunc
from zfit.util.container import convert_to_container


class SimpleFunction(BaseFunc):

    def __init__(self, func, name="Function", **parameters):
        super().__init__(name=name, **parameters)
        self._value_func = self._check_input_x_function(func)

    def _value(self, x):
        return self._value_func(x, **self.get_parameters())


class BaseFunctorFunc(BaseFunc):
    def __init__(self, funcs, name="BaseFunctorFunc", **kwargs):
        super().__init__(name=name, **kwargs)
        funcs = convert_to_container(funcs)
        self.funcs = funcs

    def _get_dependents(self):  # TODO: change recursive to `only_floating`?
        dependents = super()._get_dependents()  # get the own parameter dependents
        func_dependents = self._extract_dependents(self.funcs, only_floating=only_floating)  # flatten
        return dependents.union(func_dependents)


class SumFunc(BaseFunctorFunc):
    def __init__(self, funcs, dims=None, name="SumFunc", **kwargs):
        super().__init__(funcs=funcs, name=name, **kwargs)
        self.dims = dims

    def _value(self, x):
        sum_funcs = tf.accumulate_n([func(x) for func in self.funcs])
        return sum_funcs


class ProdFunc(BaseFunctorFunc):
    def __init__(self, funcs, name="SumFunc", **kwargs):
        super().__init__(funcs=funcs, name=name, **kwargs)

    def _value(self, x):
        value = self.funcs[0](x)
        for func in self.funcs[1:]:
            value *= func(x)
        return value

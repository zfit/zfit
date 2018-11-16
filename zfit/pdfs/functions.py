import tensorflow as tf

from zfit.core.basefunc import BaseFunc
from zfit.util.container import convert_to_container


class Function(BaseFunc):

    def __init__(self, func, name="Function", **parameters):
        super().__init__(name=name, **parameters)
        self._value_func = self._check_input_x_function(func)

    def _value(self, x):
        return self._value_func(x)


class BaseFunctorFunc(BaseFunc):
    def __init__(self, funcs, name="BaseFunctorFunc", **kwargs):
        super().__init__(name=name, **kwargs)
        funcs = convert_to_container(funcs)
        self.funcs = funcs


class SumFunc(BaseFunctorFunc):
    def __init__(self, funcs, name="SumFunc", **kwargs):
        super().__init__(funcs=funcs, name=name, **kwargs)

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

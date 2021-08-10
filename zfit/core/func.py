from zfit_interface.func import ZfitFunc
from zfit_interface.variables import ZfitParam, ZfitAxis

from zfit import Data
from zfit.core.values import ValueHolder


def to_value_holder(var):
    if not isinstance(var, ValueHolder):
        var = ValueHolder(var)
    return var


def to_data(value, space):
    data = Data.from_tensor(obs=space, tensor=value)  # TODO
    return data


class Func(ZfitFunc):
    def __init__(self, var, output_var=None, label=None):
        self.var = var
        self.params = {k: v for k, v in var.items() if isinstance(v, ZfitParam)}
        self.space = {k: v for k, v in var.items() if isinstance(v, ZfitAxis)}
        self.output_var = output_var
        self.label = label

    def __call__(self, var):
        var = to_value_holder(var)
        output = self._auto_func(var)
        if self.output_var is not None:
            output = to_data(output, self.output_var)
        return output

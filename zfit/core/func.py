#  Copyright (c) 2022 zfit
from __future__ import annotations

from contextlib import suppress

from zfit_interface.func import ZfitFunc
from zfit_interface.variables import ZfitParam, ZfitAxis

from zfit import Data
from zfit.core.values import ValueHolder
from zfit.util.exception import SpecificFunctionNotImplemented, WorkInProgressError


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

    def __call__(self, var=None):
        var = to_value_holder(var)
        output = self._auto_func(var)
        if self.output_var is not None:
            output = to_data(output, self.output_var)
        return output

    def values(self, *, var, options=None):
        return self._call_values(var=var, options=options)

    def _call_values(self, var=None, options=None):
        with suppress(SpecificFunctionNotImplemented):
            return self._values(var, options=options)  # TODO: auto_value?
        return self._fallback_values(var=var, options=options)

    def _fallback_values(self, var, norm, options):
        raise WorkInProgressError

    def _values(self, var, norm, options):
        raise SpecificFunctionNotImplemented

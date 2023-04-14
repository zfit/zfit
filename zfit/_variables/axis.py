#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Iterable

import hist
import zfit_interface as zinterface
from hist.axestuple import NamedAxesTuple

# @tfp.experimental.auto_composite_tensor()
# class Regular(hist.axis.Regular, tfp.experimental.AutoCompositeTensor, family='zfit'):
#     pass
from zfit.core.interfaces import ZfitBinning


class Variable(zinterface.variables.ZfitVar):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class SpaceV2:
    def __init__(self, axes):
        self.axes = axes

    def __getitem__(self, key):
        key = to_var_str(key)
        for axis in self.axes:
            if axis.name == key:
                return axis
        else:
            raise KeyError(f"{key} not in {self}.")

    def __iter__(self):
        yield from self.axes

    @property
    def names(self):
        return [axis.name for axis in self]


def to_var_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, zinterface.variables.ZfitVar):
        return value.name


class Axis(Variable):
    def __init__(self, name):
        super().__init__(name=name)


class UnbinnedAxis(Axis):
    def __init__(self, name, lower=None, upper=None):
        super().__init__(name)
        self.lower = lower
        self.upper = upper


# TODO: fill out below and don't just use the hist objects
class HashableAxisMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.name == "":
            raise ValueError(
                "Currently, a binning has to have a name coinciding with the obs."
            )

    def __hash__(self):
        return hash(tuple(self.edges))


class RegularBinning(HashableAxisMixin, hist.axis.Regular, ZfitBinning, family="zfit"):
    def __init__(self, bins: int, start: float, stop: float, *, name: str) -> None:
        super().__init__(bins, start, stop, name=name, flow=False)


class VariableBinning(
    HashableAxisMixin, hist.axis.Variable, ZfitBinning, family="zfit"
):
    def __init__(self, edges: Iterable[float], *, name: str) -> None:
        super().__init__(edges=edges, name=name, flow=False)


class Binnings(hist.axestuple.NamedAxesTuple):
    pass


def histaxis_to_axis(axis):
    return axis


def axis_to_histaxis(axis):
    return axis


def new_from_axis(axis):
    if isinstance(axis, hist.axis.Regular):
        lower, upper = axis.edges[0], axis.edges[-1]
        return RegularBinning(axis.size, lower, upper, name=axis.name)
    if isinstance(axis, hist.axis.Variable):
        return VariableBinning(axis.edges, name=axis.name)
    raise ValueError(f"{axis} is not a valid axis.")


def histaxes_to_binning(binnings):
    new_binnings = []
    for binning in binnings:
        new_binnings.append(new_from_axis(binning))
    return Binnings(new_binnings)


def binning_to_histaxes(binnings):
    return binnings

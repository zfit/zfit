#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable

import hist
import zfit_interface as zinterface

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
        msg = f"{key} not in {self}."
        raise KeyError(msg)

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
    return None


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
            msg = "Currently, a binning has to have a name coinciding with the obs."
            raise ValueError(msg)

    def __hash__(self):
        return hash(tuple(self.edges))


class RegularBinning(HashableAxisMixin, hist.axis.Regular, ZfitBinning, family="zfit"):
    def __init__(self, bins: int, start: float, stop: float, *, name: str) -> None:
        super().__init__(bins, start, stop, name=name, flow=False)


class VariableBinning(HashableAxisMixin, hist.axis.Variable, ZfitBinning, family="zfit"):
    def __init__(self, edges: Iterable[float], *, name: str) -> None:
        super().__init__(edges=edges, name=name, flow=False)


class Binnings(hist.axestuple.NamedAxesTuple):
    pass


HIST_BINNING_TYPES = (hist.axis.Regular, hist.axis.Variable)


def histaxis_to_axis(axis):
    return axis


def axis_to_histaxis(axis):
    return axis


def new_from_axis(axis):
    if isinstance(axis, hist.axis.Regular):
        lower, upper = axis.edges[0], axis.edges[-1]
        if axis.transform is not None:
            msg = (
                "Transformed axes are not supported. Please convert it explicitly to a Variable axis using the edges."
                "Example: ax2 = hist.axis.Variable(ax1.edges, name='x')."
                "If this is an issue or you prefer to have this automatically converted, please open an issue on github with zfit."
            )
            raise ValueError(msg)
        return RegularBinning(axis.size, lower, upper, name=axis.name)
    if isinstance(axis, hist.axis.Variable):
        return VariableBinning(axis.edges, name=axis.name)
    msg = f"{axis} is not a valid axis."
    raise ValueError(msg)


def histaxes_to_binning(binnings):
    new_binnings = []
    for binning in binnings:
        new_binnings.append(new_from_axis(binning))
    return Binnings(new_binnings)


def binning_to_histaxes(binnings):
    return binnings

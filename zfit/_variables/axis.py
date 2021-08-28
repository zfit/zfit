#  Copyright (c) 2021 zfit
import hist
import tensorflow_probability as tfp
import zfit_interface as zinterface
from hist.axestuple import NamedAxesTuple


@tfp.experimental.auto_composite_tensor()
class Regular(hist.axis.Regular, tfp.experimental.AutoCompositeTensor, family='zfit'):
    pass


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
    def __hash__(self):
        return hash(tuple(self.edges))


class Regular(HashableAxisMixin, hist.axis.Regular, family='zfit'):
    pass


class Variable(HashableAxisMixin, hist.axis.Variable, family='zfit'):
    pass


class Binning(hist.axestuple.NamedAxesTuple):
    pass


def histaxis_to_axis(axis):
    return axis


def axis_to_histaxis(axis):
    return axis


def histaxes_to_binning(binnings):
    return binnings


def binning_to_histaxes(binnings):
    return binnings

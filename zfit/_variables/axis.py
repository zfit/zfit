#  Copyright (c) 2021 zfit
import zfit_interface as zinterface


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

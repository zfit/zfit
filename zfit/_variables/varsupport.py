#  Copyright (c) 2021 zfit
import zfit_interface.variables
import tensorflow_probability as tfp

import zfit.util.container


@tfp.experimental.auto_composite_tensor()
class AxisSupports(tfp.experimental.AutoCompositeTensor):

    def __init__(self, var, *, full=None, limits=None, unbinned=None, data=None, types=None):
        types = zfit.util.container.convert_to_container(types)
        if types:
            if full or limits or unbinned or data:
                raise ValueError
        elif full:
            if limits or unbinned or data or types:
                raise ValueError

        if not isinstance(var, zfit_interface.variables.ZfitVar):
            raise TypeError(f"var has to be a ZfitVar, not {var}.")
        self.var = var
        self.full = full
        self.limits = limits
        self.unbinned = unbinned
        self.data = data
        self.types = types

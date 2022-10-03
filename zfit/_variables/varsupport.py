#  Copyright (c) 2022 zfit
from __future__ import annotations

import tensorflow_probability as tfp
import zfit_interface.variables

import zfit.util.container


@tfp.experimental.auto_composite_tensor()
class VarSupports(tfp.experimental.AutoCompositeTensor):
    def __init__(
        self,
        var,
        *,
        full=None,
        space=None,
        scalar=None,
        vectorspace=None,
        binned=None,
        data=None,
        types=None,
    ):
        types = zfit.util.container.convert_to_container(types)
        if types:
            if full or space or binned or data or scalar or vectorspace:
                raise ValueError
        elif full:
            if space or binned or data or types or scalar or vectorspace:
                raise ValueError
        elif not (space or scalar or vectorspace or binned or data):
            raise ValueError("Need to support at least something.")
        if data:
            scalar = True
        if vectorspace:
            space = True

        if not isinstance(var, zfit_interface.variables.ZfitVar):
            raise TypeError(f"var has to be a ZfitVar, not {var}.")
        self.var = var
        self.full = full or False
        self.scalar = scalar or False
        self.vectorspace = vectorspace or False
        self.space = space or False
        self.binned = binned or False
        self.data = data or False
        self.types = types or []

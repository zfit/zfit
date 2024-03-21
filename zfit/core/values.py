#  Copyright (c) 2024 zfit

from __future__ import annotations

import collections
from collections.abc import Mapping

import tensorflow_probability as tfp
from zfit_interface.variables import ZfitVar

from zfit.core.interfaces import ZfitData, ZfitParameter, ZfitSpace
from zfit.util.container import convert_to_container


@tfp.experimental.auto_composite_tensor()
class ValueHolder(tfp.experimental.AutoCompositeTensor):
    def __init__(
        self,
        args,
        variables: Mapping,
        norm: ValueHolder = None,
        target=None,
        holders=None,
    ):
        args = convert_to_container(args)
        variables = self._check_input_variables(variables)
        varmap = {name: var.name for name, var in variables.items()}
        self.target = target
        self._varmap = varmap
        self._vararg = self._create_vararg_map(args, varmap)

        # needed to create auto composite
        self.norm = norm
        self.holders = holders
        self.variables = variables
        self.args = args

    def get_var(self, name):
        if name not in self._varmap:
            msg = f"{name} is not a valid name. Has to be one of {tuple(self._varmap.keys())}"
            raise ValueError(msg)
        varname = self._varmap["name"]
        for arg in self.args:
            if isinstance(arg, ZfitVar):
                if varname == arg.name:
                    return arg
            elif isinstance(arg, ZfitData):
                if varname in arg.obs:
                    return arg[varname]
            else:
                msg = "We missed something somewhere. Please report this, it's a bug."
                raise AssertionError(msg)
        return None

    def _check_input_variables(self, variables):
        if not isinstance(variables, collections.abc.Mapping):
            msg = f"variables has to be a Mapping, not {variables}"
            raise TypeError(msg)
        not_var = {var for var in variables.values() if not isinstance(var, ZfitVar)}
        if not_var:
            msg = f"The following values in {variables} are not ZfitVar: {not_var}"
            raise TypeError(msg)

        return variables

    def __getitem__(self, item):
        try:
            index = self.names.index[item]
        except KeyError as error:
            msg = f"{self} does not contain {item}."
            raise KeyError(msg) from error
        return self.args[index]

    def __contains__(self, item):
        return item in self.names

    @property
    def params(self):
        return {k: v for k, v in zip(self.names, self.args) if isinstance(v, ZfitParameter)}

    @property
    def space(self):
        return {k: v for k, v in zip(self.names, self.args) if isinstance(v, ZfitSpace)}

    @property
    def datasets(self):
        return {k: v for k, v in zip(self.names, self.args) if isinstance(v, ZfitData)}

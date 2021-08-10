#  Copyright (c) 2021 zfit
import collections

import tensorflow_probability as tfp

from zfit.core.interfaces import ZfitParameter, ZfitSpace, ZfitData
from zfit.util.container import convert_to_container


@tfp.experimental.auto_composite_tensor()
class ValueHolder(tfp.experimental.AutoCompositeTensor):
    def __init__(self, args, names=None, target=None, holders=None):
        if isinstance(args, collections.Mapping):
            if names is not None:
                raise ValueError(f"names {names} has to be None if args {args} is a dict")
            names = args.keys()
            args = args.values()
        else:
            args = convert_to_container(args)
        self.args = args
        self.names = names
        self.target = target
        self.holders = holders

    def __getitem__(self, item):
        try:
            index = self.names.index[item]
        except KeyError as error:
            raise KeyError(f"{self} does not contain {item}.") from error
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

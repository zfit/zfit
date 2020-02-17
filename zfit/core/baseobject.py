"""Baseclass for most objects appearing in zfit."""
#  Copyright (c) 2020 zfit

from collections import OrderedDict
from typing import List

import tensorflow as tf

from zfit.core.dependents import BaseDependentsMixin
from .interfaces import ZfitObject, ZfitNumeric
from ..util import ztyping
from ..util.cache import Cachable
from ..util.container import DotDict

_COPY_DOCSTRING = """Creates a copy of the {zfit_type}.

        Note: the copy {zfit_type} may continue to depend on the original
        initialization arguments.

        Args:
          name (str):
          **overwrite_parameters: String/value dictionary of initialization
            arguments to override with new value.

        Returns:
          {zfit_type}: A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters_kwargs, i.e.,
            `dict(self.parameters, **overwrite_params)`.
        """


class BaseObject(ZfitObject):

    def __init__(self, name, **kwargs):
        assert not kwargs, "kwargs not empty, the following arguments are not captured: {}".format(kwargs)
        super().__init__()

        self._name = name  # TODO: uniquify name?

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._repr = DotDict()  # TODO: make repr more sophisticated
        # cls._repr.zfit_type = cls
        # cls.copy.__doc__ = _COPY_DOCSTRING.format(zfit_type=cls.__name__)

    @property
    def name(self) -> str:
        """The name of the object."""
        return self._name

    def copy(self, deep: bool = False, name: str = None, **overwrite_params) -> "ZfitObject":

        new_object = self._copy(deep=deep, name=name, overwrite_params=overwrite_params)
        return new_object

    def _copy(self, deep, name, overwrite_params):  # TODO(Mayou36) L: representation?
        if deep:
            raise NotImplementedError("Unfortunately, this feature is not implemented.")
        if name is None:
            name = self.name + "_copy"  # TODO: improve name mangling
        # params = self.parameters.copy()
        raise RuntimeError("This copy should not be used.")

        # params.update(overwrite_params)
        # new_object = type(self)(name=name, **params)
        # return new_object

    def __eq__(self, other: object) -> bool:
        if not isinstance(self, type(other)):
            return False
        try:
            for key, own_element in self._repr.items():
                if not own_element == other._repr.get(key):  # TODO: make repr better
                    return False
        except AttributeError:
            return self is other
        return True  # no break occurred

    def __hash__(self):
        return object.__hash__(self)


class BaseNumeric(Cachable, BaseDependentsMixin, ZfitNumeric, BaseObject):

    def __init__(self, name, params, **kwargs):
        if 'dtype' in kwargs:  # TODO(Mayou36): proper dtype handling?
            self._dtype = kwargs.pop('dtype')
        super().__init__(name=name, **kwargs)
        from zfit.core.parameter import convert_to_parameter

        params = params or OrderedDict()
        params = OrderedDict(sorted((n, convert_to_parameter(p)) for n, p in params.items()))
        self.add_cache_dependents(params)

        # parameters = OrderedDict(sorted(parameters))  # to always have a consistent order
        self._params = params
        self._repr.params = self.params

    @property
    def dtype(self) -> tf.DType:
        """The dtype of the object"""
        return self._dtype

    @property
    def params(self) -> ztyping.ParametersType:
        return self._params

    def get_params(self, only_floating: bool = False, names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
        """Return the parameters. If it is empty, automatically return all floating variables.

        Args:
            only_floating (): If True, return only the floating parameters.
            names (): The names of the parameters to return.

        Returns:
            list(`ZfitParameters`):
        """
        if isinstance(names, str):
            names = (names,)
        if names is not None:
            missing_names = set(names).difference(self.params.keys())
            if missing_names:
                raise KeyError("The following names are not valid parameter names")
            params = [self.params[name] for name in names]
        else:
            params = list(self.params.values())

        if only_floating:
            params = self._filter_floating_params(params=params)
        return params

    @staticmethod
    def _filter_floating_params(params):
        params = [param for param in params if param.floating]
        return params

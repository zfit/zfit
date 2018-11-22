import abc
from collections import OrderedDict
import itertools
from typing import List, Set

import tensorflow as tf

import zfit
from zfit.util import ztyping
from .interfaces import ZfitObject
from ..util.container import convert_to_container, DotDict


class BaseObject(ZfitObject):
    _zfit_type = "BaseObject"

    def __init__(self, name, dtype, parameters, **kwargs):
        super().__init__(**kwargs)
        self._name = name  # TODO: uniquify name?
        self._dtype = dtype
        parameters = OrderedDict(sorted(parameters))  # to always have a consistent order
        self._parameters = parameters
        self._repr.parameters = self.parameters
        self._repr.zfit_type = self._zfit_type

    def __init_subclass__(cls, **kwargs):
        cls._repr = DotDict()  # TODO: make repr more sophisticated
        cls._zfit_type = None

    @property
    def name(self) -> str:
        """The name of the object."""
        return self._name

    @property
    def dtype(self) -> tf.DType:
        """The dtype of the object"""
        return self._dtype

    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
        """Return a list of all independent :py:class:`~zfit.Parameter` that this object depends on.

        Args:
            only_floating (bool): If `True`, only return floating :py:class:`~zfit.Parameter`
        """
        dependents = self._get_dependents()
        if only_floating:
            dependents = set(filter(lambda p: p.floating, dependents))
        return dependents

    @abc.abstractmethod
    def _get_dependents(self) -> ztyping.DependentsType:
        raise NotImplementedError

    @staticmethod
    def _extract_dependents(zfit_objects: List[ZfitObject]) -> Set["zfit.Parameters"]:
        """Calls the `get_dependents` method on every object and returns a combined set.

        Args:
            zfit_objects ():

        Returns:
            set(zfit.Parameter): A set of independent Parameters
        """
        zfit_object = convert_to_container(zfit_objects)
        dependents = (obj.get_dependents() for obj in zfit_object)
        dependents = set(itertools.chain.from_iterable(dependents))  # flatten
        return dependents

    @property
    def parameters(self):
        return self._parameters

    def copy(self, deep: bool = False, **overwrite_params) -> "ZfitObject":
        """Creates a deep copy of the {zfit_type}.

        Note: the copy pdf may continue to depend on the original
        initialization arguments.

        Args:
          **override_parameters: String/value dictionary of initialization
            arguments to override with new values.

        Returns:
          pdf: A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters_kwargs, i.e.,
            `dict(self.parameters, **override_parameters_kwargs)`.
        """.format(zfit_type=self._repr.zfit_type)

    def get_parameters(self, only_floating=True, names=None) -> List['Parameter']:
        """Return the parameters. If it is empty, automatically set and return all floating variables.

        Args:
            only_floating (): If True, return only the floating parameters.
            names (): The names of the parameters to return.

        Returns:
            list(`zfit.FitParameters`):
        """
        if isinstance(names, str):
            names = (names,)
        if names is not None:
            missing_names = set(names).difference(self.parameters.keys())
            if missing_names:
                raise KeyError("The following names are not valid parameter names")
            params = [self.parameters[name] for name in names]
        else:
            params = list(self.parameters.values())

        if only_floating:
            params = self._filter_floating_params(params=params)
        return params

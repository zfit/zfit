"""Baseclass for most objects appearing in zfit."""
#  Copyright (c) 2020 zfit
import itertools
import warnings
from collections import OrderedDict
from typing import Optional, Iterable, Set

import tensorflow as tf
from ordered_set import OrderedSet

from .dependents import BaseDependentsMixin
from .interfaces import ZfitObject, ZfitNumericParametrized, ZfitParameter, ZfitParametrized, ZfitIndependentParameter
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.checks import NotSpecified
from ..util.container import DotDict, convert_to_container

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
        raise NotImplementedError("This copy should not be used.")

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


class BaseParametrized(ZfitParametrized):

    def get_params(self, floating: Optional[bool] = True, yields: Optional[bool] = None,
                   extract_independent: Optional[bool] = True, only_floating=NotSpecified) -> Set["ZfitParameter"]:
        if only_floating is not NotSpecified:
            floating = only_floating
            warnings.warn("`only_floating` is deprecated and will be removed in the future, use `floating` instead.")

        params = self.params.values()
        params = extract_filter_params(params, floating=floating, extract_independent=extract_independent)
        return params


class BaseNumeric(GraphCachable, BaseDependentsMixin, BaseParametrized, ZfitNumericParametrized, BaseObject):

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

    # def get_params(self, floating: Optional[bool] = True,
    #                yields: Optional[bool] = None,
    #                extract_independent: Optional[bool] = True,
    #                only_floating: bool = True) -> Set[ZfitParameter]:
    #     """Recursively collect parameters that this object depends on according to the filter criteria.
    #
    #     Which parameters should be included can be steered using the arguments as a filter.
    #      - **None**: do not filter on this. E.g. `floating=None` will return parameters that are floating as well as
    #         parameters that are fixed.
    #      - **True**: only return parameters that fulfil this criterion
    #      - **False**: only return parameters that do not fulfil this criterion. E.g. `floating=False` will return
    #         only parameters that are not floating.
    #
    #     Args:
    #         floating: if a parameter is floating, e.g. if :py:meth:`~ZfitParameter.floating` returns `True`
    #         yields: if a parameter is a yield of the _current_ model. This won't be applied recursively, but may include
    #            yields if they do also represent a parameter parametrizing the shape. So if the yield of the current
    #            model depends on other yields (or also non-yields), this will be included. If, however, just submodels
    #            depend on a yield (as their yield) and it is not correlated to the output of our model, they won't be
    #            included.
    #         extract_independent: If the parameter is not an independent parameter, i.e. if it is not a
    #             `ZfitIndependentParameter`, it will extract all independent parameters that dependent parameters
    #             depend on.
    #     """
    #
    #     # def get_params(self, only_floating: bool = False, names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
    #     #     """Return the parameters. If it is empty, automatically return all floating variables.
    #     #
    #     #     Args:
    #     #         only_floating (): If True, return only the floating parameters.
    #     #         names (): The names of the parameters to return.
    #     #
    #     #     Returns:
    #     #         list(`ZfitParameters`):
    #     #     """
    #     #     if isinstance(names, str):
    #     #         names = (names,)
    #     #     if names is not None:
    #     #         missing_names = set(names).difference(self.params.keys())
    #     #         if missing_names:
    #     #             raise KeyError("The following names are not valid parameter names")
    #     #         params = [self.params[name] for name in names]
    #     #     else:
    #     params = list(self.params.values())
    #
    #     if only_floating:
    #         params = self._filter_floating_params(params=params)
    #     return params

    @staticmethod
    def _filter_floating_params(params):
        params = [param for param in params if isinstance(param, ZfitIndependentParameter) and param.floating]
        return params


def extract_filter_params(params: Iterable[ZfitParametrized],
                          floating: Optional[bool] = True,
                          extract_independent: Optional[bool] = True) -> Set[ZfitParameter]:
    params = convert_to_container(params, container=OrderedSet)

    if extract_independent:
        params = OrderedSet(itertools.chain.from_iterable(param.get_params(floating=floating,
                                                                           extract_independent=True,
                                                                           yields=None)
                                                          for param in params))

    if floating is not None:
        if not extract_independent and not all(param.independent for param in params):
            raise ValueError("Since `extract_dependent` is not set to True, there are maybe dependent parameters for "
                             "which `floating` is an ill-defined attribute.")
        params = OrderedSet((p for p in params if p.floating == floating))
    return params

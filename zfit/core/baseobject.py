"""Baseclass for most objects appearing in zfit."""

#  Copyright (c) 2023 zfit

from __future__ import annotations

import contextlib
import itertools
from collections import OrderedDict
from collections.abc import Iterable

import tensorflow as tf
from ordered_set import OrderedSet

from .dependents import BaseDependentsMixin
from .interfaces import (
    ZfitIndependentParameter,
    ZfitNumericParametrized,
    ZfitObject,
    ZfitParameter,
    ZfitParametrized,
)
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.checks import NotSpecified
from ..util.container import convert_to_container
from ..util.exception import BreakingAPIChangeError


class BaseObject(ZfitObject):
    def __init__(self, name, **kwargs):
        assert (
            not kwargs
        ), f"kwargs not empty, the following arguments are not captured: {kwargs}"
        super().__init__()

        self._name = name  # TODO: uniquify name?

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._repr = {}  # TODO: make repr more sophisticated

    @property
    def name(self) -> str:
        """The name of the object."""
        return self._name

    def copy(
        self, deep: bool = False, name: str = None, **overwrite_params
    ) -> ZfitObject:
        new_object = self._copy(deep=deep, name=name, overwrite_params=overwrite_params)
        return new_object

    def _copy(self, deep, name, overwrite_params):  # TODO(Mayou36) L: representation?
        raise NotImplementedError("This copy should not be used.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(self, type(other)):
            return False
        with contextlib.suppress(AttributeError):
            for key, own_element in self._repr.items():
                if not own_element == other._repr.get(key):
                    return False
        return self is other
        # return True  # no break occurred

    def __hash__(self):
        return object.__hash__(self)


class BaseParametrized(BaseObject, ZfitParametrized):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(**kwargs)
        from zfit.core.parameter import convert_to_parameter

        params = params or OrderedDict()
        # params = OrderedDict(sorted((n, convert_to_parameter(p)) for n, p in params.items()))
        params = {n: convert_to_parameter(p) for n, p in params.items()}  # why sorted?

        # parameters = OrderedDict(sorted(parameters))  # to always have a consistent order
        self._params = params
        self._repr["params"] = self.params

    def get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        only_floating=NotSpecified,
    ) -> set[ZfitParameter]:
        """Recursively collect parameters that this object depends on according to the filter criteria.

        Which parameters should be included can be steered using the arguments as a filter.
         - **None**: do not filter on this. E.g. ``floating=None`` will return parameters that are floating as well as
            parameters that are fixed.
         - **True**: only return parameters that fulfil this criterion
         - **False**: only return parameters that do not fulfil this criterion. E.g. ``floating=False`` will return
            only parameters that are not floating.

        Args:
            floating: if a parameter is floating, e.g. if :py:meth:`~ZfitParameter.floating` returns `True`
            is_yield: if a parameter is a yield of the _current_ model. This won't be applied recursively, but may include
               yields if they do also represent a parameter parametrizing the shape. So if the yield of the current
               model depends on other yields (or also non-yields), this will be included. If, however, just submodels
               depend on a yield (as their yield) and it is not correlated to the output of our model, they won't be
               included.
            extract_independent: If the parameter is an independent parameter, i.e. if it is a ``ZfitIndependentParameter``.
        """
        if only_floating is not NotSpecified:
            raise BreakingAPIChangeError(
                "The argument `only_floating` has been renamed to `floating`."
            )
        return self._get_params(
            floating=floating,
            is_yield=is_yield,
            extract_independent=extract_independent,
        )

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
    ) -> set[ZfitParameter]:
        if (
            is_yield is True
        ):  # we want exclusively yields, we don't have them by default
            params = OrderedSet()
        else:
            params = self.params.values()
            params = extract_filter_params(
                params, floating=floating, extract_independent=extract_independent
            )
        return params

    @property
    def params(self) -> ztyping.ParameterType:
        return self._params


class BaseNumeric(
    GraphCachable,
    BaseDependentsMixin,
    BaseParametrized,
    ZfitNumericParametrized,
    BaseObject,
):
    def __init__(self, **kwargs):
        if "dtype" in kwargs:  # TODO(Mayou36): proper dtype handling?
            self._dtype = kwargs.pop("dtype")
        super().__init__(**kwargs)
        self.add_cache_deps(self.params.values())

    @property
    def dtype(self) -> tf.DType:
        """The dtype of the object."""
        return self._dtype

    @staticmethod
    def _filter_floating_params(params):
        params = [
            param
            for param in params
            if isinstance(param, ZfitIndependentParameter) and param.floating
        ]
        return params


def extract_filter_params(
    params: Iterable[ZfitParametrized],
    floating: bool | None = True,
    extract_independent: bool | None = True,
) -> set[ZfitParameter]:
    params = convert_to_container(params, container=OrderedSet)

    if extract_independent is not False:
        params_indep = OrderedSet(
            itertools.chain.from_iterable(
                param.get_params(
                    floating=floating, extract_independent=True, is_yield=None
                )
                for param in params
            )
        )
        if extract_independent is True:
            params = params_indep
        else:  # None
            params |= params_indep
    if floating is not None:
        if not extract_independent and not all(param.independent for param in params):
            raise ValueError(
                "Since `extract_dependent` is not set to True, there are maybe dependent parameters for "
                "which `floating` is an ill-defined attribute."
            )
        params = OrderedSet(p for p in params if p.floating == floating)
    return params

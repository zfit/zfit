"""Baseclass for most objects appearing in zfit."""

#  Copyright (c) 2024 zfit

from __future__ import annotations

import contextlib
import itertools
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Union

import tensorflow as tf
from ordered_set import OrderedSet

from ..minimizers.interface import ZfitResult
from ..util import ztyping
from ..util.cache import GraphCachable
from ..util.container import convert_to_container
from ..util.exception import ParamNameNotUniqueError
from .interfaces import (
    ZfitIndependentParameter,
    ZfitNumericParametrized,
    ZfitObject,
    ZfitParameter,
    ZfitParametrized,
)


class BaseObject(ZfitObject):
    def __init__(self, name, **kwargs):
        assert not kwargs, f"kwargs not empty, the following arguments are not captured: {kwargs}"
        super().__init__()

        self._name = name  # TODO: uniquify name?

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._repr = {}  # TODO: make repr more sophisticated

    @property
    def name(self) -> str:
        """The name of the object."""
        return self._name

    def copy(self, deep: bool = False, name: str | None = None, **overwrite_params) -> ZfitObject:
        return self._copy(deep=deep, name=name, overwrite_params=overwrite_params)

    def _copy(self, deep, name, overwrite_params):  # noqa: ARG002
        msg = "This copy should not be used."
        raise NotImplementedError(msg)

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


def convert_param_values(params: Union[Mapping[Union[str, ztyping.ParamType], float], ZfitResult]):
    """Convert the mapping or `ZfitResult` to a dictionary of str -> value.

    Args:
        params: A mapping of parameter names to values or a `ZfitResult`.

    Returns:
        A dictionary of parameter names to values.

    Raises:
        TypeError: If `params` is not a mapping or a `ZfitResult`.
    """
    if params is None:
        params = {}
    elif isinstance(params, ZfitResult):
        params = params.values

    elif not isinstance(params, Mapping):
        msg = f"`params` has to be a mapping (dict-like) or a `ZfitResult`, is {params} of type {type(params)}."
        raise TypeError(msg)
    return {param.name if isinstance(param, ZfitParameter) else param: value for param, value in params.items()}


class BaseParametrized(BaseObject, ZfitParametrized):
    def __init__(self, params, autograd_params=None, **kwargs) -> None:
        super().__init__(**kwargs)
        from zfit.core.parameter import convert_to_parameter

        params = params or {}
        params = {n: convert_to_parameter(p) for n, p in params.items()}  # why sorted?

        if autograd_params is None:
            autograd_params = True
        if autograd_params is True:
            autograd_params = (*params, "yield")
        elif isinstance(autograd_params, Iterable):
            nonexisting_params = set(autograd_params) - set(params)
            nonexisting_params -= {"yield"}  # yield is not part of params, so needs to be removed
            if nonexisting_params:
                msg = f"Parameters {nonexisting_params} are not in the parameters of {self}: {params}."
                raise ValueError(msg)
        else:
            msg = f"Invalid value for `autograd_params`: {autograd_params}"
            raise ValueError(msg)

        self._autograd_params = autograd_params

        # parameters = dict(sorted(parameters))  # to always have a consistent order
        self._params = params
        self._repr["params"] = self.params
        # check if the object has duplicated names as parameters

    def _assert_params_unique(self):
        """Assert that the parameters are unique, i.e. no parameter has the same name as another one.

        Raises:
            ValueError: If the parameters are not unique.
        """
        all_params = self.get_params(floating=None, is_yield=None, extract_independent=None)  # get **all** params
        counted_names = Counter(param.name for param in all_params)
        if duplicated_names := {name for name, count in counted_names.items() if count > 1}:  # set comprehension
            msg = (
                f"The following parameter names appear more than once in {self}: {duplicated_names}."
                f"This is new behavior: before, a parameter with the same name could not exists, now it can."
                f"However, they are not allowed to be within the same function/PDF/loss, as this would result in ill-defined behavior."
            )
            raise ParamNameNotUniqueError(msg)

    def get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        *,
        autograd: bool | None = None,
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
            extract_independent: If the parameter is an independent parameter,
                i.e. if it is a ``ZfitIndependentParameter``.
        """
        if autograd is True:
            allparams = self._get_params(
                floating=floating, is_yield=is_yield, extract_independent=extract_independent, autograd=None
            )
            nograd_params = self._get_params(
                floating=None, is_yield=None, extract_independent=None, autograd=False
            )  # we want to get *any* param that is not autograd
            params = allparams - nograd_params
        else:
            params = self._get_params(
                floating=floating, is_yield=is_yield, extract_independent=extract_independent, autograd=autograd
            )
        return params

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        assert autograd is not True, "This should never be True, it's only for internal use."
        if is_yield is True:  # we want exclusively yields, we don't have them by default
            params = OrderedSet()
        else:
            params = []
            for name, p in self.params.items():
                # we either collect _all_ params or only the ones that do not support autograd
                if autograd is None or (autograd is False and name not in self._autograd_params):
                    params.append(p)

            params = extract_filter_params(params, floating=floating, extract_independent=extract_independent)
        return params

    @property
    def params(self) -> ztyping.ParameterType:
        return self._params

    @contextlib.contextmanager
    def _check_set_input_params(self, params, guarantee_checked=None):
        paramvalues = self._check_convert_input_paramvalues(params, guarantee_checked)
        import zfit

        with zfit.param.set_values(tuple(paramvalues.keys()), tuple(paramvalues.values())):
            yield paramvalues

    def _check_convert_input_paramvalues(self, params, guarantee_checked=None) -> dict[str, float] | None:
        if guarantee_checked is None:
            guarantee_checked = False

        newpars = {}
        if params is not None:
            if guarantee_checked:
                newpars = params
            else:
                params = convert_param_values(params)
                newpars = {}
                all_params = self.get_params(floating=None, is_yield=None)
                toset_params = params.copy()
                for param in all_params:
                    if (pname := param.name) in params:
                        newpars[param] = toset_params.pop(pname)

                if toset_params:
                    msg = f"Parameters {toset_params} were not found in the parameters of {self}: {all_params}."
                    raise ValueError(msg)

                # This is for converting and passing through, complicated?
                # for param in all_params:
                #     if param in params or param.name in params:
                #         newpars[param] = params[param]
                #     else:
                #         newpars[param] = znp.asarray(param.value())

        return newpars


class BaseNumeric(
    GraphCachable,
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
        return [param for param in params if isinstance(param, ZfitIndependentParameter) and param.floating]


def extract_filter_params(
    params: Iterable[ZfitParametrized] | ZfitParametrized,
    floating: bool | None = True,
    extract_independent: bool | None = True,
) -> OrderedSet[ZfitParameter]:
    params = convert_to_container(params, container=OrderedSet)

    if extract_independent is not False:
        params_indep = OrderedSet(
            itertools.chain.from_iterable(
                param.get_params(floating=floating, is_yield=None, extract_independent=True) for param in params
            )
        )
        if extract_independent is True:
            params = params_indep
        else:  # None
            params |= params_indep
    if floating is not None:
        if not extract_independent and not all(param.independent for param in params):
            msg = (
                "Since `extract_dependent` is not set to True, there are maybe dependent parameters for "
                "which `floating` is an ill-defined attribute."
            )
            raise ValueError(msg)
        params = OrderedSet(p for p in params if p.floating == floating)
    return params

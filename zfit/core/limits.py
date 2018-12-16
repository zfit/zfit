"""
.. include:: ../../docs/subst_types.rst

# NamedSpace and limits

Limits define a certain interval in a specific dimension. This can be used to define, for example,
the limits of an integral over several dimensions.

There are two ways of creating a :py:class:`NamedSpace`, either from the limits or from the boundaries
(which are arbitrary definitions here).

### by dimensions (from limits)

Shortcut: if only a 1-dim tuple is given, it is assumed to be the limits from a 1-dim interval.

If the limits in each dimension are known, this is the easiest way to construct a :py:class:`NamedSpace`.
A simple example to represent the limits for the first dim from 1 to 4 and for the second dim
from -1 to 4 *and* from 6 to 8:
>>> limits_as_tuple = ((1, 4), (-1, 4, 6, 8))  # 2 dimensions
>>> limits = NamedSpace.from_limits(limits=limits_as_tuple, axes=(0, 1))  # which dimensions the limits correspond to

This can be retrieved in the same way with
>>> limits.get_limits()

General form: ((lower1_dim1, upper1_dim1, lower2_dim1, upper2_dim1,...),
(lower1_dim2, upper1_dim2, lower2_dim2, upper2_dim2,...), ...

The tuples don't have to have the same length!

### by intervals (from boundaries)

The disadvantage of the previous method is that only rectangular limits are covered
(in the sense that you can't specify e.g. 1-dim: from 1 to 2 and second dim from 4 to 6 AND 1-dim
from 10 to 12 and 2.-dim from 50 to 52, so creating individual patches in the multidimensional space).
Therefore a different way of specifying limits is possible, basically by defining chunks of the
lower and the upper limits. The shape is (n_limits, n_dims).

Example: 1-dim: 1 to 4, 2.-dim: 21 to 24 AND 1.-dim: 6 to 7, 2.-dim 26 to 27
>>> lower = ((1, 21), (6, 26))
>>> upper = ((4, 24), (7, 27))
>>> limits2 = NamedSpace.from_boundaries(lower=lower, upper=upper, axes=(0, 1))

General form:

lower = ((lower1_dim1, lower1_dim2, lower1_dim3), (lower2_dim1, lower2_dim2, lower2_dim3),...)
upper = ((upper1_dim1, upper1_dim2, upper1_dim3), (upper2_dim1, upper2_dim2, upper2_dim3),...)

## Using :py:class:`NamedSpace`

:py:class:`NamedSpace` offers a few useful functions to easier deal with the intervals

### Handling areas

For example when doing a MC integration using the expectation value, it is mandatory to know
the total area of your intervals. You can retrieve the total area or (if multiple limits (=intervals
 are given) the area of each interval.

 >>> area = limits2.areas
 >>> area_1, area_2 = limits2.area_by_boundaries(rel=False)  # if rel is True, return the fraction of 1


### Convert and retrieve the limits

The limits can be converted from the "by dimensions" form to "by intervals" and vice-versa, though
the latter will raise an error if no save conversion is possible! (e.g. in our example above,
limits2 converted limits "by dimension" will raise an error). So retrieving limits should be done via
>>> lower, upper = limits2.limits

which you can now iterate through. For example, to calc an integral (assuming there is a function
`integrate` taking the lower and upper limits and returning the function), you can do
>>> def integrate(lower_limit, upper_limit): return 42  # dummy function
>>> integral = sum(integrate(lower_limit=low, upper_limit=up) for low, up in zip(lower, upper))
"""
from collections import OrderedDict
import copy
import functools
import inspect
from typing import Tuple, Union, List, Optional, Iterable, Callable, Dict

import numpy as np

from zfit.core.baseobject import BaseObject
from zfit.core.interfaces import ZfitNamedSpace
from zfit.util import ztyping
from zfit.util.checks import NOT_SPECIFIED
from zfit.util.container import convert_to_container
from zfit.util.exception import (NormRangeNotImplementedError, MultipleLimitsNotImplementedError, AxesNotSpecifiedError,
                                 ObsNotSpecifiedError, OverdefinedError, IntentionNotUnambiguousError,
                                 LimitsUnderdefinedError, )
from zfit.util.temporary import TemporarilySet


# Singleton


class NamedSpace(ZfitNamedSpace, BaseObject):
    AUTO_FILL = object()
    ANY = object()
    ANY_LOWER = object()  # TODO: need different upper, lower?
    ANY_UPPER = object()
    _limit_replace = {'lower': {None: ANY_LOWER},
                      'upper': {None: ANY_UPPER}}

    def __init__(self, obs: ztyping.ObsTypeInput, limits: Optional[ztyping.LimitsTypeInput] = None,
                 name: Optional[str] = "NamedSpace"):
        """Define a space with the name (`obs`) of the axes (and it's number) and possibly it's limits.

        Args:
            obs (str, List[str,...]):
            limits ():
            name (str):
        """
        obs = self._check_convert_input_obs(obs, allow_not_spec=True)

        if name is None:
            name = "NamedSpace_" + "_".join(obs)
        super().__init__(name=name)
        self._axes = None
        self._obs = obs
        self._check_set_limits(limits=limits)

    @classmethod
    def _from_any(cls, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                  limits: Optional[ztyping.LimitsTypeInput] = None,
                  name: str = None) -> "NamedSpace":
        if obs is None:
            new_space = NamedSpace.from_axes(axes=axes, limits=limits, name=name)
        else:
            new_space = NamedSpace(obs=obs, limits=limits, name=name)
            new_space._axes = axes

        return new_space

    @classmethod
    def from_axes(cls, axes: ztyping.AxesTypeInput,
                  limits: Optional[ztyping.LimitsTypeInput] = None,
                  name: str = None) -> "NamedSpace":
        """Create a space from `axes` instead of from `obs`.

        Args:
            axes ():
            limits ():
            name (str):

        Returns:
            NamedSpace
        """
        axes = convert_to_container(value=axes, container=tuple)
        if axes is None:
            raise AxesNotSpecifiedError("Axes cannot be `None`")
        fake_obs = (str(axis) for axis in axes)  # in order to create an instance

        new_space = cls(obs=fake_obs, limits=limits, name=name)
        new_space._obs = None
        new_space._axes = new_space._check_convert_input_axes(axes)
        return new_space

    @staticmethod
    def _convert_obs_to_str(obs):
        if isinstance(obs, NamedSpace):
            obs = obs.obs
        else:
            obs = convert_to_container(obs, container=tuple)
        return obs

    def _convert_axes_to_int(self, axes):
        if isinstance(axes, NamedSpace):
            axes = axes.axes
        else:
            axes = convert_to_container(axes, container=tuple)
        return axes

    def _check_set_limits(self, limits: ztyping.LimitsTypeInput):

        if limits is not None and limits is not False:
            lower, upper = limits
            limits = self._check_convert_input_lower_upper(lower=lower, upper=upper)
        self._limits = limits

    def _check_convert_input_lower_upper(self, lower, upper):
        replace_lower = self._limit_replace.get('lower', {})
        replace_upper = self._limit_replace.get('upper', {})

        lower = self._check_convert_input_limit(limit=lower, replace=replace_lower)
        upper = self._check_convert_input_limit(limit=upper, replace=replace_upper)
        lower_is_iterable = lower is not None or lower is not False
        upper_is_iterable = upper is not None or upper is not False
        if not (lower_is_iterable or upper_is_iterable) and lower is not upper:
            ValueError("Lower and upper limits wrong:"
                       "\nlower = {l}"
                       "\nupper = {u}".format(l=lower, u=upper))
        if lower_is_iterable ^ upper_is_iterable:
            raise ValueError("Lower and upper limits wrong:"
                             "\nlower = {l}"
                             "\nupper = {u}".format(l=lower, u=upper))
        if lower_is_iterable and upper_is_iterable:
            if not np.shape(lower) == np.shape(upper) or len(np.shape(lower)) != 2:
                raise ValueError("Lower and/or upper limits invalid:"
                                 "\nlower: {lower}"
                                 "\nupper: {upper}".format(lower=lower, upper=upper))

            if not np.shape(lower)[1] == self.n_obs:
                raise ValueError("Limits shape not compatible with number of obs/axes"
                                 "\nlower: {lower}"
                                 "\nupper: {upper}"
                                 "\nn_obs: {n_obs}".format(lower=lower, upper=upper, n_obs=self.n_obs))
        return lower, upper

    def _check_convert_input_limit(self, limit: Union[ztyping.LowerTypeInput, ztyping.UpperTypeInput],
                                   replace=None) -> Union[ztyping.LowerTypeReturn, ztyping.UpperTypeReturn]:
        """Check and sanitize the lower or upper limit.

        Args:
            limit ():

        Returns:

        """
        replace = {} if replace is None else replace
        if limit is NOT_SPECIFIED or limit is None:
            return None
        # TODO: allow automatically correct shape?
        if np.shape(limit) == ():
            limit = ((limit,),)
        if np.shape(limit[0]) == ():
            raise ValueError("Shape of limit {} wrong.".format(limit))
        #     limit = (limit,)

        # replace
        if replace:
            limit = tuple(tuple(replace.get(l, l) for l in lim) for lim in limit)

        return limit

    def _check_set_lower_upper(self, lower: ztyping.LowerTypeInput, upper: ztyping.UpperTypeInput):

        if lower is None or lower is False:
            limits = lower
        else:
            lower = self._check_convert_input_limit(lower)
            upper = self._check_convert_input_limit(upper)
            limits = lower, upper
        self._check_set_limits(limits=limits)

    def _check_convert_input_axes(self, axes: ztyping.AxesTypeInput,
                                  allow_none: bool = False,
                                  allow_not_spec: bool = False) -> ztyping.AxesTypeReturn:
        if axes is NOT_SPECIFIED:
            if allow_not_spec:
                return None
            else:
                raise AxesNotSpecifiedError("TODO: Cannot be NOT_SPECIFIED")
        if axes is None:
            if allow_none:
                return None
            else:
                raise AxesNotSpecifiedError("TODO: Cannot be None")
        axes = convert_to_container(value=axes, container=tuple)
        return axes

    def _check_convert_input_obs(self, obs: ztyping.ObsTypeInput,
                                 allow_none: bool = False,
                                 allow_not_spec: bool = False) -> ztyping.ObsTypeReturn:
        """Input check: Convert `NOT_SPECIFIED` to None or check if obs are all strings.

        Args:
            obs (str, List[str], None, NOT_SPECIFIED):

        Returns:
            type:
        """
        if obs is NOT_SPECIFIED:
            if allow_not_spec:
                return None
            else:
                raise ObsNotSpecifiedError("TODO: Cannot be NOT_SPECIFIED")
        if obs is None:
            if allow_none:
                return None
            else:
                raise ObsNotSpecifiedError("TODO: Cannot be None")

        if isinstance(obs, NamedSpace):
            obs = obs.obs
        else:
            obs = convert_to_container(obs, container=tuple)
            obs_not_str = tuple(o for o in obs if not isinstance(o, str))
            if obs_not_str:
                raise ValueError("The following observables are not strings: {}".format(obs_not_str))
        return obs

    @property
    def limits(self) -> ztyping.LimitsTypeReturn:
        """Return the limits.

        Returns:

        """
        return self._limits

    @property
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits.

        Returns:

        """
        limits = self.limits
        if limits is None or limits is False:
            return limits
        else:
            return limits[0]

    @property
    def upper(self) -> ztyping.UpperTypeReturn:
        """Return the upper limits.

        Returns:

        """
        limits = self.limits
        if limits is None or limits is False:
            return limits
        else:
            return self.limits[1]

    @property
    def n_obs(self) -> int:  # TODO(naming): better name? Like rank?
        """Return the number of observables/axes.

        Returns:
            int >= 1
        """

        if self.obs is None:
            length = len(self.axes)
        else:
            length = len(self.obs)
        return length

    @property
    def n_limits(self) -> int:
        """The number of different limits.

        Returns:
            int >= 1
        """
        if self.lower is None or self.lower is False:
            return 0
        return len(self.lower)

    @property
    def obs(self) -> ztyping.ObsTypeReturn:
        """The observables ("axes with str")the space is defined in.

        Returns:

        """
        return self._obs

    @property
    def axes(self) -> ztyping.AxesTypeReturn:
        """The axes ("obs with int") the space is defined in.

        Returns:

        """
        return self._axes

    def get_axes(self, obs: ztyping.ObsTypeInput = None,
                 as_dict: bool = False,
                 autofill: bool = False) -> Union[ztyping.AxesTypeReturn, Dict[str, int]]:
        """Return the axes corresponding to the `obs` (or all if None).

        Args:
            obs ():
            as_dict (bool): If True, returns a ordered dictionary with {obs: axis}
            autofill (bool): If True and the axes are not specified, automatically fill
                them with the default numbering and return (not setting them).

        Returns:
            Tuple, OrderedDict

        Raises:
            ValueError: if the requested `obs` do not match with the one defined in the range
            AxesNotSpecifiedError: If the axes in this `Space` have not been specified.
        """
        # check input
        obs = self._check_convert_input_obs(obs=obs, allow_none=True)
        axes = self.axes
        if axes is None:
            if autofill:
                axes = tuple(range(self.n_obs))
            else:
                raise AxesNotSpecifiedError("The axes have not been specified")

        if obs is not None:
            try:
                axes = tuple(axes[self.obs.index(o)] for o in obs)
            except KeyError:
                missing_obs = set(obs) - set(self.obs)
                raise ValueError("The requested observables {mis} are not contained in the defined "
                                 "observables {obs}".format(mis=missing_obs, obs=self.obs))
        else:
            obs = self.obs
        if as_dict:
            axes = OrderedDict((o, ax) for o, ax in zip(obs, axes))

        return axes

    def iter_limits(self, as_tuple: bool = True) -> ztyping._IterLimitsTypeReturn:
        """Return the limits, either as `Space` objects or as pure limits-tuple.

        This makes iterating over limits easier: `for limit in space.iter_limits()`
        allows to, for example, pass `limit` to a function that can deal with simple limits
        only or if `as_tuple` is True the `limit` can be directly used to calculate something.

        Example:
            ```python
            for lower, upper in space.iter_limits(as_tuple=True):
                integrals = integrate(lower, upper)  # calculate integral
            integral = sum(integrals)
            ```

        Returns:
            List[NamedSpace] or List[limit,...]:
        """
        if as_tuple:
            return list(zip(self.lower, self.upper))
        else:
            space_objects = []
            for lower, upper in self.iter_limits(as_tuple=True):
                if not (lower is None or lower is False):
                    lower = (lower,)
                    upper = (upper,)
                    limit = lower, upper
                space = NamedSpace._from_any(obs=self.obs, axes=self.axes, limits=limit)
                space_objects.append(space)
            return space_objects

    def with_limits(self, limits: ztyping.LimitsTypeInput, name: Optional[str] = None) -> "NamedSpace":
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits ():
            name (str):

        Returns:
            NamedSpace
        """
        new_space = self.copy(limits=limits, name=name)
        return new_space

    def with_obs(self, obs: ztyping.ObsTypeInput) -> "NamedSpace":
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `NamedSpace`
        """
        obs = self._convert_obs_to_str(obs)
        new_indices = self.get_reorder_indices(obs=obs)
        new_space = self.copy()
        new_space.reorder_by_indices(indices=new_indices)
        return new_space

    def with_axes(self, axes: ztyping.AxesTypeInput) -> "NamedSpace":
        """Sort by `obs` and return the new instance.

        Args:
            axes ():

        Returns:
            `NamedSpace`
        """
        # TODO: what if self.axes is None? Just add them?
        axes = self._convert_axes_to_int(axes)
        new_indices = self.get_reorder_indices(axes=axes)
        new_space = self.copy()
        new_space.reorder_by_indices(indices=new_indices)
        return new_space

    def get_reorder_indices(self, obs: ztyping.ObsTypeInput = None,
                            axes: ztyping.AxesTypeInput = None) -> Tuple[int]:
        """Indices that would order `self.obs` as `obs` respectively `self.axes` as `axes`.

        Args:
            obs ():
            axes ():

        Returns:

        """
        obs_none = obs is None
        axes_none = axes is None
        # if not obs_none ^ axes_none:
        #     raise ValueError("Exactly one of `obs` _or_ `axes` have to be specified.")

        # raise ObsNotSpecifiedError("Observables `obs` not specified in this space.")
        # if self.axes is None and not axes_none:
        #     raise AxesNotSpecifiedError("Axes`axes` not specified in this space.")
        obs_is_defined = self.obs is not None and not obs_none
        axes_is_defined = self.axes is not None and not axes_none
        if not (obs_is_defined or axes_is_defined):
            raise ValueError("Neither the `obs` nor `axes` are defined.")

        if obs_is_defined:
            old, new = self.obs, obs
        else:
            old, new = self.axes, axes

        new_indices = _reorder_indices(old=old, new=new)
        return new_indices

    def reorder_by_indices(self, indices: Tuple[int]):
        """Reorder (temporarily) the `Space` with the given indices.

        Args:
            indices ():

        """
        old_indices = [None] * len(indices)
        for old_i, new_i in enumerate(indices):
            old_indices[new_i] = old_i

        def setter(indices):
            self._reorder_limits(indices=indices)
            self._reorder_axes(indices=indices)
            self._reorder_obs(indices=indices)

        def getter():
            return old_indices

        return TemporarilySet(value=indices, setter=setter, getter=getter)

    def _reorder_limits(self, indices: Tuple[int], inplace: bool = True) -> ztyping.LimitsTypeReturn:
        limits = self.limits
        if limits is not None and limits is not False:
            # print("DEBUG,limits", limits)
            lower, upper = limits
            lower = tuple(tuple(lower[i] for i in indices) for lower in lower)
            upper = tuple(tuple(upper[i] for i in indices) for upper in upper)
            limits = lower, upper
        # print("DEBUG 2,limits", limits)

        if inplace:
            self._limits = limits
        return limits

    def _reorder_axes(self, indices: Tuple[int], inplace: bool = True) -> ztyping.AxesTypeReturn:
        axes = self.axes
        if axes is not None:
            axes = tuple(axes[i] for i in indices)
        if inplace:
            self._axes = axes
        return axes

    def _reorder_obs(self, indices: Tuple[int], inplace: bool = True) -> ztyping.ObsTypeReturn:
        obs = self.obs
        if obs is not None:
            obs = tuple(obs[i] for i in indices)
        if inplace:
            self._obs = obs
        return obs

    def get_obs_axes(self):
        if self.obs is None:
            raise ObsNotSpecifiedError("Obs not specified, cannot create `obs_axes`")
        if self.axes is None:
            raise AxesNotSpecifiedError("Axes not specified, cannot create `obs_axes`")

        # creaet obs_axes dict
        obs_axes = OrderedDict((o, ax) for o, ax in zip(self.obs, self.axes))
        return obs_axes

    def _set_obs_axes(self, obs_axes: Union[ztyping.OrderedDict[str, int], Dict[str, int]], ordered: bool = False):
        """(Reorder) set the observables and the `axes`.

        Temporarily if used with a context manager.

        Args:
            obs_axes (OrderedDict[str, int]): An ordered dict with {obs: axes}.

        Returns:

        """
        if ordered and not isinstance(obs_axes, OrderedDict):
            raise IntentionNotUnambiguousError("`ordered` is True but not an `OrderedDict` was given."
                                               "Error due to safety (in Python <3.7, dicts are not guaranteed to be"
                                               "ordered).")
        if ordered:
            if self.obs is not None:
                permutation_index = tuple(
                    self.obs.index(o) for o in obs_axes if o in self.obs)  # the future index of the space
            elif self.axes is not None:
                permutation_index = tuple(
                    self.axes.index(ax) for ax in obs_axes.values() if ax in self.axes)  # the future index of the space
            else:
                assert False, "This should never be reached."
            limits = self._reorder_limits(indices=permutation_index, inplace=False)
            obs = tuple(obs_axes.keys())
            axes = tuple(obs_axes.values())
        else:
            if self.obs is not None:
                obs = self.obs
                axes = tuple(obs_axes[o] for o in obs)
            elif self.axes is not None:
                axes = self.axes
                axes_obs = {v: k for k, v in obs_axes.items()}
                obs = tuple(axes_obs[ax] for ax in axes)
            else:
                raise ValueError("Either `obs` or `axes` have to be specified if the `obs_axes` dict"
                                 "is not ordered and `ordered` is False.")
            limits = self.limits

        value = limits, obs, axes

        def setter(arguments):
            limits, obs, axes = arguments

            self._check_set_limits(limits=limits)
            self._obs = obs
            self._axes = axes

        def getter():
            return self.limits, self.obs, self.axes

        return TemporarilySet(value=value, setter=setter, getter=getter)

    def with_obs_axes(self, obs_axes: Union[ztyping.OrderedDict[str, int], Dict[str, int]],
                      ordered: bool = False) -> "NamedSpace":
        """Return a new `Space` with reordered observables and set the `axes`.


        Args:
            ordered (bool): If True (and the `obs_axes` is an `OrderedDict`), the
            obs_axes (OrderedDict[str, int]): An ordered dict with {obs: axes}.

        Returns:
            NamedSpace:
        """
        with self._set_obs_axes(obs_axes=obs_axes, ordered=ordered):
            return copy.deepcopy(self)

    def with_autofill_axes(self, overwrite: bool = False) -> "NamedSpace":
        """Return a `Space` with filled axes corresponding to range(len(n_obs)).

        Args:
            overwrite (bool): If `self.axes` is not None, replace the axes with the autofilled ones.
                If axes is already set, don't do anything if `overwrite` is False.

        Returns:
            `NamedSpace`
        """
        if self.axes is None or overwrite:
            new_axes = tuple(range(self.n_obs))
            new_space = self.copy(axes=new_axes)
        else:
            new_space = self

        return new_space

    def area(self) -> float:
        """Return the total area of all the limits and axes. Useful, for example, for MC integration."""
        return sum(self.iter_areas(rel=False))

    def iter_areas(self, rel: bool = False) -> Tuple[float, ...]:
        """Return the areas of each interval

        Args:
            rel (bool): If True, return the relative fraction of each interval
        Returns:
            Tuple[float]:
        """
        areas = self._calculate_areas(limits=self.limits)
        if rel:
            areas = tuple(np.linalg.norm(areas, ord=1))
        return areas

    @staticmethod
    @functools.lru_cache()
    def _calculate_areas(limits) -> Tuple[float]:
        areas = tuple(float(np.prod(np.array(up) - np.array(low))) for low, up in zip(*limits))
        return areas

    def get_subspace(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                     name: Optional[str] = None) -> "NamedSpace":
        """Create a `Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs (str, Tuple[str]):
            axes (int, Tuple[int]):
            name ():

        Returns:

        """
        if obs is not None and axes is not None:
            raise ValueError("Cannot specify `obs` *and* `axes` to get subspace.")
        if axes is None and obs is None:
            raise ValueError("Either `obs` or `axes` has to be specified and not None")

        # try to use observables to get index
        obs = self._check_convert_input_obs(obs=obs, allow_none=True)
        if obs is not None:
            try:
                sub_index = tuple(self.obs.index(o) for o in obs)
            except KeyError:
                raise KeyError("Cannot get subspace from `obs` {} as this observables are not defined"
                               "in this space.".format({set(obs) - set(self.obs)}))
            except AttributeError:
                raise ObsNotSpecifiedError("Observables have not been specified in this space.")

        # try to use axes to get index
        axes = self._check_convert_input_axes(axes=axes, allow_none=True)
        if axes is not None:
            try:
                sub_index = tuple(self.axes.index(ax) for ax in axes)
            except KeyError:
                raise KeyError("Cannot get subspace from `axes` {} as this axes are not defined"
                               "in this space.".format({set(axes) - set(self.axes)}))
            except AttributeError:
                raise AxesNotSpecifiedError("Axes have not been specified for this space.")

        sub_obs = self.obs if self.obs is None else tuple(self.obs[i] for i in sub_index)
        sub_axes = self.axes if self.axes is None else tuple(self.axes[i] for i in sub_index)

        # use index to get limits
        limits = self.limits
        if limits is None or limits is False:
            sub_limits = limits
        else:
            lower, upper = limits
            sub_lower = tuple(tuple(lim[i] for i in sub_index) for lim in lower)
            sub_upper = tuple(tuple(lim[i] for i in sub_index) for lim in upper)
            sub_limits = sub_lower, sub_upper

        new_space = NamedSpace._from_any(obs=sub_obs, axes=sub_axes, limits=sub_limits, name=name)

        return new_space

    # Operators

    def copy(self, name: Optional[str] = None, **overwrite_kwargs) -> "NamedSpace":
        """Create a new `Space` using the current attributes and overwriting with `overwrite_overwrite_kwargs`.

        Args:
            name (str): The new name. If not given, the new instance will be named the same as the
                current one.
            **overwrite_kwargs ():

        Returns:
            NamedSpace
        """
        name = self.name if name is None else name

        kwargs = {'name': name,
                  'limits': self.limits,
                  'axes': self.axes,
                  'obs': self.obs}
        kwargs.update(overwrite_kwargs)
        if set(overwrite_kwargs) - set(kwargs):
            raise KeyError("Not usable keys in `overwrite_kwargs`: {}".format(set(overwrite_kwargs) - set(kwargs)))

        new_space = NamedSpace._from_any(**kwargs)
        return new_space

    def __le__(self, other):  # TODO: refactor for boundaries
        if not isinstance(other, type(self)):
            return NotImplemented
        axes_not_none = self.axes is not None and other.axes is not None
        obs_not_none = self.obs is not None and other.obs is not None
        if not (axes_not_none or obs_not_none):  # if both are None
            return False
        if axes_not_none:
            if set(self.axes) != set(other.axes):
                return False

        if obs_not_none:
            if set(self.obs) != set(other.obs):
                return False

        # if not (obs_not_none or o)
        # axes_obs_not_same = self.axes != other.axes or self.obs != other.obs
        # if obs_set_not_same or axes_set_not_same or axes_obs_not_same:
        #     return False

        # check limits
        if self.limits is None:
            if other.limits is None:
                return True
            else:
                return False

        elif self.limits is False:
            if other.limits is False:
                return True
            else:
                return False

        # if self.obs is None:
        #     reorder_axes = self.axes
        # else:
        #     reorder_axes = None
        reorder_indices = other.get_reorder_indices(obs=self.obs, axes=self.axes)
        with other.reorder_by_indices(reorder_indices):

            # check explicitely if they match
            # for each limit in self, find another matching in other
            for lower, upper in self.iter_limits(as_tuple=True):
                limit_is_le = False
                for other_lower, other_upper in other.iter_limits(as_tuple=True):
                    # each entry *has to* match the entry of the other limit, otherwise it's not the same
                    for low, up, other_low, other_up in zip(lower, upper, other_lower, other_upper):
                        axis_le = 0  # False
                        # a list of `or` conditions
                        axis_le += other_low == low and up == other_up  # TODO: approx limit comparison?
                        axis_le += other_low == low and other_up is self.ANY_UPPER  # TODO: approx limit
                        # comparison?
                        axis_le += other_low is self.ANY_LOWER and up == other_up  # TODO: approx limit
                        # comparison?
                        axis_le += other_low is self.ANY_LOWER and other_up is self.ANY_UPPER
                        if not axis_le:  # if not the same, don't test other dims
                            break
                    else:
                        limit_is_le = True  # no break -> all axes coincide
                if not limit_is_le:  # for this `limit`, no other_limit matched
                    return False
        return True

    def __ge__(self, other):
        return other.__le__(self)

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented

        is_eq = True
        # if self.obs is None or other.obs is None:
        #     is_eq *= self.axes == other.axes
        # elif self.axes is None or other.axes is None:
        #     is_eq *= self.obs == other.obs
        # else:
        is_eq *= self.obs == other.obs
        is_eq *= self.axes == other.axes
        is_eq *= self.limits == other.limits
        return bool(is_eq)

    def __hash__(self):
        lower = self.lower
        upper = self.upper
        if not (lower is None or lower is False):  # we want to be non-restrictive: it's just a hash, not the __eq__
            lower = frozenset(frozenset(lim) for lim in lower)
        if not (upper is None or upper is False):
            upper = frozenset(frozenset(lim) for lim in upper)

        return hash((lower, upper))


# TODO(Mayou36): extend this function and set what is allowed and what not, allow for simple limits?
def convert_to_space(obs: Optional[ztyping.ObsTypeInput] = None, axes: Optional[ztyping.AxesTypeInput] = None,
                     limits: Optional[ztyping.LimitsTypeInput] = None,
                     *, none_is_any=False, overwrite_limits: bool = False, one_dim_limits_only: bool = True,
                     simple_limits_only: bool = True) -> Union[None, NamedSpace, bool]:
    """Convert *limits* to a Space object if not already None or False.

    Args:
        obs (Union[Tuple[float, float], zfit.core.limits.NamedSpace]):
        limits ():
        axes ():
        overwrite_limits (bool): If `obs` or `axes` is a `NamedSpace` _and_ `limits` are given, return an instance
            of `NamedSpace` with the new limits. If the flag is `False`, the `limits` argument will be ignored if
        one_dim_limits_only (bool):
        simple_limits_only (bool):

    Returns:
        Union[NamedSpace, False, None]:

    Raises:
        OverdefinedError: if `obs` or `axes` is a `NamedSpace` and `axes` respectively `obs` is not `None`.
    """
    space = None

    # Test if already `NamedSpace` and handle
    if isinstance(obs, NamedSpace):
        if axes is not None:
            raise OverdefinedError("if `obs` is a `NamedSpace`, `axes` cannot be defined.")
        space = obs
    elif isinstance(axes, NamedSpace):
        if obs is not None:
            raise OverdefinedError("if `axes` is a `NamedSpace`, `obs` cannot be defined.")
        space = axes
    elif isinstance(limits, NamedSpace):
        return limits
    if space is not None:
        # set the limits if given
        if limits is not None and (overwrite_limits or space.limits is None):
            if isinstance(limits, NamedSpace):  # figure out if compatible if limits is `NamedSpace`
                if not (limits.obs == space.obs or
                        (limits.axes == space.axes and limits.obs is None and space.obs is None)):
                    raise IntentionNotUnambiguousError(
                        "`obs`/`axes` is a `NamedSpace` as well as the `limits`, but the "
                        "obs/axes of them do not match")
                else:
                    limits = limits.limits

            space = space.with_limits(limits=limits)
        return space

    # space is None again
    if not (obs is None and axes is None):
        # check if limits are allowed
        space = NamedSpace._from_any(obs=obs, axes=axes, limits=limits)  # create and test if valid
        if one_dim_limits_only and space.n_obs > 1 and space.limits:
            raise LimitsUnderdefinedError(
                "Limits more sophisticated than 1-dim cannot be auto-created from tuples. Use `NamedSpace` instead.")
        if simple_limits_only and space.limits and space.n_limits > 1:
            raise LimitsUnderdefinedError("Limits with multiple limits cannot be auto-created"
                                          " from tuples. Use `NamedSpace` instead.")
    return space


def _reorder_indices(old: Union[List, Tuple], new: Union[List, Tuple]) -> Tuple[int]:
    new_indices = tuple(old.index(o) for o in new)
    return new_indices


def no_norm_range(func):
    """Decorator: Catch the 'norm_range' kwargs. If not None, raise NormRangeNotImplementedError."""
    parameters = inspect.signature(func).parameters
    keys = list(parameters.keys())
    if 'norm_range' in keys:
        norm_range_index = keys.index('norm_range')
    else:
        norm_range_index = None

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        norm_range = kwargs.get('norm_range')
        if isinstance(norm_range, NamedSpace):
            norm_range_not_false = not (norm_range.limits is None or norm_range.limits is False)
        else:
            norm_range_not_false = not (norm_range is None or norm_range is False)
        if norm_range_index is not None:
            norm_range_is_arg = len(args) > norm_range_index
        else:
            norm_range_is_arg = False
            kwargs.pop('norm_range', None)  # remove if in signature (= norm_range_index not None)
        if norm_range_not_false or norm_range_is_arg:
            raise NormRangeNotImplementedError()
        else:
            return func(*args, **kwargs)

    return new_func


def no_multiple_limits(func):
    """Decorator: Catch the 'limits' kwargs. If it contains multiple limits, raise MultipleLimitsNotImplementedError."""
    parameters = inspect.signature(func).parameters
    keys = list(parameters.keys())
    if 'limits' in keys:
        limits_index = keys.index('limits')
    else:
        return func  # no limits as parameters -> no problem

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        limits_is_arg = len(args) > limits_index
        if limits_is_arg:
            limits = args[limits_index]
        else:
            limits = kwargs['limits']

        if limits.n_limits > 1:
            raise MultipleLimitsNotImplementedError
        else:
            return func(*args, **kwargs)

    return new_func


def supports(*, norm_range: bool = False, multiple_limits: bool = False) -> Callable:
    """Decorator: Add (mandatory for some methods) on a method to control what it can handle.

    If any of the flags is set to False, it will check the arguments and, in case they match a flag
    (say if a *norm_range* is passed while the *norm_range* flag is set to `False`), it will
    raise a corresponding exception (in this example a `NormRangeNotImplementedError`) that will
    be catched by an earlier function that knows how to handle things.

    Args:
        norm_range (bool): If False, no norm_range argument will be passed through resp. will be `None`
        multiple_limits (bool): If False, only simple limits are to be expected and no iteration is
            therefore required.
    """
    decorator_stack = []
    if not multiple_limits:
        decorator_stack.append(no_multiple_limits)
    if not norm_range:
        decorator_stack.append(no_norm_range)

    def create_deco_stack(func):
        for decorator in reversed(decorator_stack):
            func = decorator(func)
        func.__wrapped__ = supports
        return func

    return create_deco_stack

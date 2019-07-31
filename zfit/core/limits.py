"""
.. include:: ../../docs/subst_types.rst

# NamedSpace and limits

Limits define a certain interval in a specific dimension. This can be used to define, for example,
the limits of an integral over several dimensions or a normalization range.



### with limits

Therefore a different way of specifying limits is possible, basically by defining chunks of the
lower and the upper limits. The shape of a lower resp. upper limit is (n_limits, n_obs).

Example: 1-dim: 1 to 4, 2.-dim: 21 to 24 AND 1.-dim: 6 to 7, 2.-dim 26 to 27
>>> lower = ((1, 21), (6, 26))
>>> upper = ((4, 24), (7, 27))
>>> limits2 = Space(limits=(lower, upper), obs=('obs1', 'obs2')

General form:

lower = ((lower1_dim1, lower1_dim2, lower1_dim3), (lower2_dim1, lower2_dim2, lower2_dim3),...)
upper = ((upper1_dim1, upper1_dim2, upper1_dim3), (upper2_dim1, upper2_dim2, upper2_dim3),...)

## Using :py:class:`~zfit.Space`

:py:class:`NamedSpace` offers a few useful functions to easier deal with the intervals

### Handling areas

For example when doing a MC integration using the expectation value, it is mandatory to know
the total area of your intervals. You can retrieve the total area or (if multiple limits (=intervals
 are given) the area of each interval.

 >>> area = limits2.area()
 >>> area_1, area_2 = limits2.iter_areas(rel=False)  # if rel is True, return the fraction of 1


### Retrieve the limits

>>> lower, upper = limits2.limits

which you can now iterate through. For example, to calc an integral (assuming there is a function
`integrate` taking the lower and upper limits and returning the function), you can do
>>> def integrate(lower_limit, upper_limit): return 42  # dummy function
>>> integral = sum(integrate(lower_limit=low, upper_limit=up) for low, up in zip(lower, upper))
"""
#  Copyright (c) 2019 zfit

# TODO(Mayou36): update docs above

from collections import OrderedDict
import copy
import functools
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .dimension import add_spaces, combine_spaces
from .baseobject import BaseObject
from .interfaces import ZfitSpace
from ..util import ztyping
from ..util.checks import NOT_SPECIFIED
from ..util.container import convert_to_container
from ..util.exception import (AxesNotSpecifiedError, IntentionNotUnambiguousError, LimitsUnderdefinedError,
                              MultipleLimitsNotImplementedError, NormRangeNotImplementedError, ObsNotSpecifiedError,
                              OverdefinedError, LimitsNotSpecifiedError, )
from ..util.temporary import TemporarilySet


# Singleton
class Any:
    _singleton_instance = None

    def __new__(cls, *args, **kwargs):
        instance = cls._singleton_instance
        if instance is None:
            instance = super().__new__(cls)
            cls._singleton_instance = instance

        return instance

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._singleton_instance = None  # each subclass is a singleton of "itself"

    def __repr__(self):
        return '<Any>'

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    # def __eq__(self, other):
    #     return True

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return True

    # def __hash__(self):
    #     return


class AnyLower(Any):
    def __repr__(self):
        return '<Any Lower Limit>'

    # def __eq__(self, other):
    #     return False

    def __ge__(self, other):
        return False

    def __gt__(self, other):
        return False


class AnyUpper(Any):
    def __repr__(self):
        return '<Any Upper Limit>'

    # def __eq__(self, other):
    #     return False

    def __le__(self, other):
        return False

    def __lt__(self, other):
        return False


ANY = Any()
ANY_LOWER = AnyLower()
ANY_UPPER = AnyUpper()


class Space(ZfitSpace, BaseObject):
    AUTO_FILL = object()
    ANY = ANY
    ANY_LOWER = ANY_LOWER  # TODO: needed? or move everything inside?
    ANY_UPPER = ANY_UPPER

    def __init__(self, obs: ztyping.ObsTypeInput, limits: Optional[ztyping.LimitsTypeInput] = None,
                 name: Optional[str] = "Space"):
        """Define a space with the name (`obs`) of the axes (and it's number) and possibly it's limits.

        Args:
            obs (str, List[str,...]):
            limits ():
            name (str):
        """
        obs = self._check_convert_input_obs(obs)

        if name is None:
            name = "Space_" + "_".join(obs)
        super().__init__(name=name)
        self._axes = None
        self._obs = obs
        self._check_set_limits(limits=limits)

    @classmethod
    def _from_any(cls, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                  limits: Optional[ztyping.LimitsTypeInput] = None,
                  name: str = None) -> "zfit.Space":
        if obs is None:
            new_space = cls.from_axes(axes=axes, limits=limits, name=name)
        else:
            new_space = cls(obs=obs, limits=limits, name=name)
            new_space._axes = axes

        return new_space

    @classmethod
    def from_axes(cls, axes: ztyping.AxesTypeInput,
                  limits: Optional[ztyping.LimitsTypeInput] = None,
                  name: str = None) -> "zfit.Space":
        """Create a space from `axes` instead of from `obs`.

        Args:
            axes ():
            limits ():
            name (str):

        Returns:
            :py:class:`~zfit.Space`
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
        if isinstance(obs, Space):
            obs = obs.obs
        else:
            obs = convert_to_container(obs, container=tuple)
        return obs

    @staticmethod
    def _convert_axes_to_int(axes):
        if isinstance(axes, Space):
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

        lower = self._check_convert_input_limit(limit=lower)
        upper = self._check_convert_input_limit(limit=upper)
        lower_is_iterable = lower is not None or lower is not False
        upper_is_iterable = upper is not None or upper is not False
        if not (lower_is_iterable or upper_is_iterable) and lower is not upper:
            ValueError("Lower and upper limits wrong:"
                       "\nlower = {lower}"
                       "\nupper = {upper}".format(lower=lower, upper=upper))
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
        if (isinstance(limit, tuple) and limit == ()) or (isinstance(limit, np.ndarray) and limit.size == 0):
            raise ValueError("Currently, () is not supported as limits. Should this be default for None?")
        if np.shape(limit) == ():
            limit = ((limit,),)
        if np.shape(limit[0]) == ():
            raise ValueError("Shape of limit {} wrong.".format(limit))

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
                                  allow_none: bool = False) -> ztyping.AxesTypeReturn:
        if axes is None:
            if allow_none:
                return None
            else:
                raise AxesNotSpecifiedError("TODO: Cannot be None")
        axes = convert_to_container(value=axes, container=tuple)  # TODO(Mayou36): extend like _check_obs?
        return axes

    def _check_convert_input_obs(self, obs: ztyping.ObsTypeInput,
                                 allow_none: bool = False) -> ztyping.ObsTypeReturn:
        """Input check: Convert `NOT_SPECIFIED` to None or check if obs are all strings.

        Args:
            obs (str, List[str], None, NOT_SPECIFIED):

        Returns:
            type:
        """
        if obs is None:
            if allow_none:
                return None
            else:
                raise ObsNotSpecifiedError("TODO: Cannot be None")

        if isinstance(obs, Space):
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
    def limit1d(self) -> Tuple[float, float]:
        """Simplified limits getter for 1 obs, 1 limit only: return the tuple(lower, upper).

        Returns:
            tuple(float, float): so :code:`lower, upper = space.limit1d` for a simple, 1 obs limit.

        Raises:
            RuntimeError: if the conditions (n_obs or n_limits) are not satisfied.
        """
        if self.n_obs > 1:
            raise RuntimeError("Cannot call `limit1d, as `Space` has more than one observables: {}".format(self.n_obs))
        if self.n_limits > 1:
            raise RuntimeError("Cannot call `limit1d, as `Space` has several limits: {}".format(self.n_limits))

        limits = self.limits
        if limits in (None, False):
            limit = limits
        else:
            (lower,), (upper,) = limits
            limit = lower[0], upper[0]
        return limit

    @property
    def limit2d(self) -> Tuple[float, float, float, float]:
        """Simplified `limits` for exactly 2 obs, 1 limit: return the tuple(low_obs1, low_obs2, up_obs1, up_obs2).

        Returns:
            tuple(float, float, float, float): so `low_x, low_y, up_x, up_y = space.limit2d` for a single, 2 obs limit.
                low_x is the lower limit in x, up_x is the upper limit in x etc.

        Raises:
            RuntimeError: if the conditions (n_obs or n_limits) are not satisfied.
        """
        if self.n_obs != 2:
            raise RuntimeError("Cannot call `limit2d, as `Space` has not two observables: {}".format(self.n_obs))
        if self.n_limits > 1:
            raise RuntimeError("Cannot call `limit2d, as `Space` has several limits: {}".format(self.n_limits))

        limits = self.limits
        if limits in (None, False):
            limit = limits
        else:
            (lower,), (upper,) = limits
            limit = *lower, *upper
        return limit

    @property
    def limits1d(self) -> Tuple[float]:
        """Simplified `.limits` for exactly 1 obs, n limits: return the tuple(low_1, ..., low_n, up_1, ..., up_n).

        Returns:
            tuple(float, float, ...): so `low_1, low_2, up_1, up_2 = space.limits1d` for several, 1 obs limits.
                low_1 to up_1 is the first interval, low_2 to up_2 is the second interval etc.

        Raises:
            RuntimeError: if the conditions (n_obs or n_limits) are not satisfied.
        """
        if self.n_obs > 1:
            raise RuntimeError("Cannot call `limits1d, as `Space` has more than one observable: {}".format(self.n_obs))
        # if self.n_limits > 1:
        #     raise RuntimeError("Cannot call `limit1d, as `Space` has several limits: {}".format(self.n_limits))

        limits = self.limits
        if limits in (None, False):
            limit = limits
        else:
            new_lower, new_upper = [], []
            for lower, upper in self.iter_limits(as_tuple=True):
                new_lower.append(lower[0])
                new_upper.append(upper[0])
            new_lower = tuple(new_lower)
            new_upper = tuple(new_upper)
            limit = *new_lower, *new_upper
        return limit

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
            AxesNotSpecifiedError: If the axes in this :py:class:`~zfit.Space` have not been specified.
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
        """Return the limits, either as :py:class:`~zfit.Space` objects or as pure limits-tuple.

        This makes iterating over limits easier: `for limit in space.iter_limits()`
        allows to, for example, pass `limit` to a function that can deal with simple limits
        only or if `as_tuple` is True the `limit` can be directly used to calculate something.

        Example:
            .. code:: python

                for lower, upper in space.iter_limits(as_tuple=True):
                    integrals = integrate(lower, upper)  # calculate integral
                integral = sum(integrals)


        Returns:
            List[:py:class:`~zfit.Space`] or List[limit,...]:
        """
        if not self.limits:
            raise LimitsNotSpecifiedError("Space does not have limits, cannot iterate over them.")
        if as_tuple:
            return tuple(zip(self.lower, self.upper))
        else:
            space_objects = []
            for lower, upper in self.iter_limits(as_tuple=True):
                if not (lower is None or lower is False):
                    lower = (lower,)
                    upper = (upper,)
                    limit = lower, upper
                else:
                    limit = lower
                space = type(self)._from_any(obs=self.obs, axes=self.axes, limits=limit)
                space_objects.append(space)
            return tuple(space_objects)

    def with_limits(self, limits: ztyping.LimitsTypeInput, name: Optional[str] = None) -> "zfit.Space":
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits ():
            name (str):

        Returns:
            :py:class:`~zfit.Space`
        """
        new_space = self.copy(limits=limits, name=name)
        return new_space

    def with_obs(self, obs: ztyping.ObsTypeInput) -> "zfit.Space":
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            :py:class:`~zfit.Space`
        """
        if obs is None or obs == self.obs:
            return self
        obs = self._convert_obs_to_str(obs)
        new_indices = self.get_reorder_indices(obs=obs)
        new_space = self.reorder_by_indices(indices=new_indices)
        return new_space

    def with_axes(self, axes: ztyping.AxesTypeInput) -> "zfit.Space":
        """Sort by `obs` and return the new instance.

        Args:
            axes ():

        Returns:
            :py:class:`~zfit.Space`
        """
        # TODO: what if self.axes is None? Just add them?
        if axes is None or axes == self.axes:
            return self
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

        obs_is_defined = self.obs is not None and not obs_none
        axes_is_defined = self.axes is not None and not axes_none
        if not (obs_is_defined or axes_is_defined):
            raise ValueError("Neither the `obs` nor `axes` are defined.")

        if obs_is_defined:
            old, new = self.obs, [o for o in obs if o in self.obs]
        else:
            old, new = self.axes, [a for a in axes if a in self.axes]

        new_indices = _reorder_indices(old=old, new=new)
        return new_indices

    def reorder_by_indices(self, indices: Tuple[int]):
        """Return a :py:class:`~zfit.Space` reordered by the indices.

        Args:
            indices ():

        """

        new_space = self.copy()
        new_space._reorder_limits(indices=indices)
        new_space._reorder_axes(indices=indices)
        new_space._reorder_obs(indices=indices)

        return new_space

    def _reorder_limits(self, indices: Tuple[int], inplace: bool = True) -> ztyping.LimitsTypeReturn:
        limits = self.limits
        if limits is not None and limits is not False:

            lower, upper = limits
            lower = tuple(tuple(lower[i] for i in indices) for lower in lower)
            upper = tuple(tuple(upper[i] for i in indices) for upper in upper)
            limits = lower, upper

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

    def get_obs_axes(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None):
        if self.obs is None:
            raise ObsNotSpecifiedError("Obs not specified, cannot create `obs_axes`")
        if self.axes is None:
            raise AxesNotSpecifiedError("Axes not specified, cannot create `obs_axes`")

        obs = self._check_convert_input_obs(obs, allow_none=True)
        axes = self._check_convert_input_axes(axes, allow_none=True)
        if obs is not None and axes is not None:
            raise OverdefinedError("Cannot use `obs` and `axes` to define which subset to access.")
        obs = self.obs if obs is None else obs
        axes = self.axes if axes is None else axes
        # only membership testing below
        obs = frozenset(obs)
        axes = frozenset(axes)

        # create obs_axes dict
        obs_axes = OrderedDict((o, ax) for o, ax in self.obs_axes.items() if o in obs or ax in axes)
        return obs_axes

    @property
    def obs_axes(self):
        # TODO(Mayou36): what if axes is None?
        return OrderedDict((o, ax) for o, ax in zip(self.obs, self.axes))

    def _set_obs_axes(self, obs_axes: Union[ztyping.OrderedDict[str, int], Dict[str, int]], ordered: bool = False,
                      allow_subset=False):
        """(Reorder) set the observables and the `axes`.

        Temporarily if used with a context manager.

        Args:
            obs_axes (OrderedDict[str, int]): An (ordered) dict with {obs: axes}.
            allow_subset ():

        Returns:

        """
        if ordered and not isinstance(obs_axes, OrderedDict):
            raise IntentionNotUnambiguousError("`ordered` is True but not an `OrderedDict` was given."
                                               "Error due to safety (in Python <3.7, dicts are not guaranteed to be"
                                               "ordered).")
        tmp_obs = self.obs if self.obs is not None else obs_axes.keys()
        self_obs_set = frozenset(tmp_obs)
        tmp_axes = self.axes if self.axes is not None else obs_axes.values()
        self_axes_set = frozenset(tmp_axes)
        if ordered:
            if self.obs is not None:
                # if not frozenset(obs_axes.keys()) <= self_obs_set:
                #     raise ValueError("TODO observables not contained")
                if not allow_subset and frozenset(obs_axes.keys()) < self_obs_set:
                    raise ValueError("subset not allowed but `obs` is only a subset of `self.obs`")
                permutation_index = tuple(
                    self.obs.index(o) for o in obs_axes if o in self_obs_set)  # the future index of the space
                self_axes_set = set(obs_axes[o] for o in self.obs if o in obs_axes)
            elif self.axes is not None:
                if not frozenset(obs_axes.values()) <= self_axes_set:
                    raise ValueError("TODO axes not contained")
                if not allow_subset and frozenset(obs_axes.values()) < self_axes_set:
                    raise ValueError("subset not allowed but `axes` is only a subset of `self.axes`")
                permutation_index = tuple(
                    self.axes.index(ax) for ax in obs_axes.values() if
                    ax in self_axes_set)  # the future index of the space
                self_obs_set = set(o for o, ax in obs_axes.items() if ax in self.axes)
            else:
                assert False, "This should never be reached."
            limits = self._reorder_limits(indices=permutation_index, inplace=False)
            obs = tuple(o for o in obs_axes.keys() if o in self_obs_set)
            axes = tuple(ax for ax in obs_axes.values() if ax in self_axes_set)
        else:
            if self.obs is not None:
                if not allow_subset and frozenset(obs_axes.keys()) < self_obs_set:
                    raise ValueError("subset not allowed TODO")
                obs = self.obs
                axes = tuple(obs_axes[o] for o in obs)
            elif self.axes is not None:
                if not allow_subset and frozenset(obs_axes.values()) < self_axes_set:
                    raise ValueError("subset not allowed TODO")
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

            self._obs = obs
            self._axes = axes
            self._check_set_limits(limits=limits)

        def getter():
            return self.limits, self.obs, self.axes

        return TemporarilySet(value=value, setter=setter, getter=getter)

    def with_obs_axes(self, obs_axes: Union[ztyping.OrderedDict[str, int], Dict[str, int]], ordered: bool = False,
                      allow_subset=False) -> "zfit.Space":
        """Return a new :py:class:`~zfit.Space` with reordered observables and set the `axes`.


        Args:
            obs_axes (OrderedDict[str, int]): An ordered dict with {obs: axes}.
            ordered (bool): If True (and the `obs_axes` is an `OrderedDict`), the
            allow_subset ():

        Returns:
            :py:class:`~zfit.Space`:
        """
        new_space = type(self)._from_any(obs=self.obs, axes=self.axes, limits=self.limits)
        new_space._set_obs_axes(obs_axes=obs_axes, ordered=ordered, allow_subset=allow_subset)
        return new_space

    def with_autofill_axes(self, overwrite: bool = False) -> "zfit.Space":
        """Return a :py:class:`~zfit.Space` with filled axes corresponding to range(len(n_obs)).

        Args:
            overwrite (bool): If `self.axes` is not None, replace the axes with the autofilled ones.
                If axes is already set, don't do anything if `overwrite` is False.

        Returns:
            :py:class:`~zfit.Space`
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
            areas = np.array(areas)
            areas /= areas.sum()
            areas = tuple(areas)
        return areas

    @staticmethod
    @functools.lru_cache()
    def _calculate_areas(limits) -> Tuple[float]:
        areas = tuple(float(np.prod(np.array(up) - np.array(low))) for low, up in zip(*limits))
        return areas

    def get_subspace(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None,
                     name: Optional[str] = None) -> "zfit.Space":
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

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
            except ValueError as error:
                print("Original message: ", error)
                raise KeyError("Cannot get subspace from `obs` {} as this observables are not defined"
                               "in this space. Only {} is defined.".format(set(obs) - set(self.obs), set(self.obs)))
            except AttributeError:  # `obs` is None -> has not attribute `index`
                raise ObsNotSpecifiedError("Observables have not been specified in this space.")

        # try to use axes to get index
        axes = self._check_convert_input_axes(axes=axes, allow_none=True)
        if axes is not None:
            try:
                sub_index = tuple(self.axes.index(ax) for ax in axes)

            except ValueError as error:
                print("Original message: ", error)
                raise KeyError("Cannot get subspace from `axes` {} as this axes are not defined"
                               "in this space. Only the following axes are {}"
                               "".format(set(axes) - set(self.axes), self.axes))
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

        new_space = type(self)._from_any(obs=sub_obs, axes=sub_axes, limits=sub_limits, name=name)

        return new_space

    # Operators

    def copy(self, name: Optional[str] = None, **overwrite_kwargs) -> "zfit.Space":
        """Create a new :py:class:`~zfit.Space` using the current attributes and overwriting with
        `overwrite_overwrite_kwargs`.

        Args:
            name (str): The new name. If not given, the new instance will be named the same as the
                current one.
            **overwrite_kwargs ():

        Returns:
            :py:class:`~zfit.Space`
        """
        name = self.name if name is None else name

        kwargs = {'name': name,
                  'limits': self.limits,
                  'axes': self.axes,
                  'obs': self.obs}
        kwargs.update(overwrite_kwargs)
        if set(overwrite_kwargs) - set(kwargs):
            raise KeyError("Not usable keys in `overwrite_kwargs`: {}".format(set(overwrite_kwargs) - set(kwargs)))

        new_space = type(self)._from_any(**kwargs)
        return new_space

    def __le__(self, other):
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

        reorder_indices = other.get_reorder_indices(obs=self.obs, axes=self.axes)
        other = other.reorder_by_indices(reorder_indices)

        # check explicitly if they match
        # for each limit in self, find another matching in other
        for lower, upper in self.iter_limits(as_tuple=True):
            limit_is_le = False
            for other_lower, other_upper in other.iter_limits(as_tuple=True):
                # each entry *has to* match the entry of the other limit, otherwise it's not the same
                for low, up, other_low, other_up in zip(lower, upper, other_lower, other_upper):
                    axis_le = 0  # False
                    axis_le += other_low == low and up == other_up  # TODO: approx limit comparison?
                    axis_le += other_low == low and other_up is ANY_UPPER  # TODO: approx limit
                    # comparison?
                    axis_le += other_low is ANY_LOWER and up == other_up  # TODO: approx limit
                    # comparison?
                    axis_le += other_low is ANY_LOWER and other_up is ANY_UPPER
                    if not axis_le:  # if not the same, don't test other dims
                        break
                else:
                    limit_is_le = True  # no break -> all axes coincide
            if not limit_is_le:  # for this `limit`, no other_limit matched
                return False
        return True

    def add(self, other: ztyping.SpaceOrSpacesTypeInput):
        """Add the limits of the spaces. Only works for the same obs.

        In case the observables are different, the order of the first space is taken.

        Args:
            other (:py:class:`~zfit.Space`):

        Returns:
            :py:class:`~zfit.Space`:
        """
        other = convert_to_container(other, container=list)
        new_space = add_spaces([self] + other)
        return new_space

    def combine(self, other: ztyping.SpaceOrSpacesTypeInput) -> ZfitSpace:
        """Combine spaces with different obs (but consistent limits).

        Args:
            other (:py:class:`~zfit.Space`):

        Returns:
            :py:class:`~zfit.Space`:
        """
        other = convert_to_container(other, container=list)
        new_space = combine_spaces([self] + other)
        return new_space

    def __add__(self, other):
        if not isinstance(other, ZfitSpace):
            raise TypeError("Cannot add a {} and a {}".format(type(self), type(other)))
        return self.add(other)

    def __mul__(self, other):
        if not isinstance(other, ZfitSpace):
            raise TypeError("Cannot combine a {} and a {}".format(type(self), type(other)))
        return self.combine(other)

    def __ge__(self, other):
        return other.__le__(self)

    def __eq__(self, other):
        if not isinstance(self, type(other)):  # TODO(Mayou36): what is a proper comparison?
            return NotImplemented

        is_eq = True
        is_eq *= self.obs == other.obs
        is_eq *= self.axes == other.axes or self.axes is None or other.axes is None
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


def convert_to_space(obs: Optional[ztyping.ObsTypeInput] = None, axes: Optional[ztyping.AxesTypeInput] = None,
                     limits: Optional[ztyping.LimitsTypeInput] = None,
                     *, overwrite_limits: bool = False, one_dim_limits_only: bool = True,
                     simple_limits_only: bool = True) -> Union[None, Space, bool]:
    """Convert *limits* to a :py:class:`~zfit.Space` object if not already None or False.

    Args:
        obs (Union[Tuple[float, float], :py:class:`~zfit.Space`]):
        limits ():
        axes ():
        overwrite_limits (bool): If `obs` or `axes` is a :py:class:`~zfit.Space` _and_ `limits` are given, return an instance
            of :py:class:`~zfit.Space` with the new limits. If the flag is `False`, the `limits` argument will be
            ignored if
        one_dim_limits_only (bool):
        simple_limits_only (bool):

    Returns:
        Union[:py:class:`~zfit.Space`, False, None]:

    Raises:
        OverdefinedError: if `obs` or `axes` is a :py:class:`~zfit.Space` and `axes` respectively `obs` is not `None`.
    """
    space = None

    # Test if already `Space` and handle
    if isinstance(obs, Space):
        if axes is not None:
            raise OverdefinedError("if `obs` is a `Space`, `axes` cannot be defined.")
        space = obs
    elif isinstance(axes, Space):
        if obs is not None:
            raise OverdefinedError("if `axes` is a `Space`, `obs` cannot be defined.")
        space = axes
    elif isinstance(limits, Space):
        return limits
    if space is not None:
        # set the limits if given
        if limits is not None and (overwrite_limits or space.limits is None):
            if isinstance(limits, Space):  # figure out if compatible if limits is `Space`
                if not (limits.obs == space.obs or
                        (limits.axes == space.axes and limits.obs is None and space.obs is None)):
                    raise IntentionNotUnambiguousError(
                        "`obs`/`axes` is a `Space` as well as the `limits`, but the "
                        "obs/axes of them do not match")
                else:
                    limits = limits.limits

            space = space.with_limits(limits=limits)
        return space

    # space is None again
    if not (obs is None and axes is None):
        # check if limits are allowed
        space = Space._from_any(obs=obs, axes=axes, limits=limits)  # create and test if valid
        if one_dim_limits_only and space.n_obs > 1 and space.limits:
            raise LimitsUnderdefinedError(
                "Limits more sophisticated than 1-dim cannot be auto-created from tuples. Use `Space` instead.")
        if simple_limits_only and space.limits and space.n_limits > 1:
            raise LimitsUnderdefinedError("Limits with multiple limits cannot be auto-created"
                                          " from tuples. Use `Space` instead.")
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
        if isinstance(norm_range, Space):
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


def convert_to_obs_str(obs):
    """Convert `obs` to the list of obs, also if it is a :py:class:`~zfit.Space`.

    """
    obs = convert_to_container(value=obs, container=tuple)
    new_obs = []
    for ob in obs:
        if isinstance(ob, Space):
            new_obs.extend(ob.obs)
        else:
            new_obs.append(ob)
    return new_obs

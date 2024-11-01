#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor

from .serialmixin import SerializableMixin

if TYPE_CHECKING:
    import zfit

import functools
import inspect
import itertools
import warnings
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress

import numpy as np
import tensorflow as tf

import zfit
import zfit.z.numpy as znp

from .. import z
from .._variables.axis import Binnings, RegularBinning, histaxes_to_binning
from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args, deprecated_norm_range
from ..util.exception import (
    AxesIncompatibleError,
    AxesNotSpecifiedError,
    BreakingAPIChangeError,
    CannotConvertToNumpyError,
    CoordinatesIncompatibleError,
    CoordinatesUnderdefinedError,
    IllegalInGraphModeError,
    IntentionAmbiguousError,
    InvalidLimitSubspaceError,
    LimitsIncompatibleError,
    LimitsNotSpecifiedError,
    LimitsUnderdefinedError,
    MultipleLimitsNotImplemented,
    NormNotImplemented,
    NumberOfEventsIncompatibleError,
    ObsIncompatibleError,
    ObsNotSpecifiedError,
    OverdefinedError,
    ShapeIncompatibleError,
    SpaceIncompatibleError,
)
from .baseobject import BaseObject
from .coordinates import (
    Coordinates,
    _convert_obs_to_str,
    convert_to_axes,
    convert_to_obs_str,
)
from .dimension import common_axes, common_obs, limits_overlap
from .interfaces import (
    ZfitData,
    ZfitLimit,
    ZfitOrderableDimensional,
    ZfitPDF,
    ZfitSpace,
)


class LimitRangeDefinition:
    pass


# Singleton
class Any(LimitRangeDefinition):
    _singleton_instance = None

    def __new__(cls, *_, **__):
        instance = cls._singleton_instance
        if instance is None:
            instance = super().__new__(cls)
            cls._singleton_instance = instance

        return instance

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._singleton_instance = None  # each subclass is a singleton of "itself"

    def __repr__(self):
        return "<Any>"

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return True


class AnyLower(Any):
    def __repr__(self):
        return "<Any Lower Limit>"


class AnyUpper(Any):
    def __repr__(self):
        return "<Any Upper Limit>"


ANY = Any()
ANY_LOWER = AnyLower()
ANY_UPPER = AnyUpper()


class V1Space:
    def __init__(self, space: ZfitSpace):
        self.space = space

    @property
    def lower(self) -> np.ndarray:
        return znp.atleast_1d(self.space.lower[0])

    @property
    def upper(self) -> np.ndarray:
        return znp.atleast_1d(self.space.upper[0])

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper

    @property
    def volume(self) -> np.ndarray:
        return znp.atleast_1d(self.space._legacy_area())

    def area(self):
        return znp.atleast_1d(self.space.area())  # intentional, we want the deprecated one to trigger


class V0Space:
    def __init__(self, space: ZfitSpace):
        self.space = space

    @property
    def lower(self):
        return self.space.lower

    @property
    def upper(self):
        return self.space.upper

    @property
    def limits(self):
        return self.space.limits

    @property
    def volume(self):
        return self.space._legacy_area()

    def area(self):
        self.space.area()  # intentional, we want the deprecated one to trigger


class VectorizeLimits:
    def __init__(self, space):
        self.space = space

    @property
    def lower(self):
        return self.space.v1.lower[None, :]

    @property
    def upper(self):
        return self.space.v1.upper[None, :]

    @property
    def limits(self):
        return znp.stack([self.lower, self.upper], axis=0)

    @property
    def volume(self):
        return self.space.v1.volume[None, :]


# TODO(warning): set a changeable warning system in zfit
def fail_not_rect(func):
    def wrapped_func(*args, **kwargs):
        self = args[0]
        if self.has_limits and not self.has_rect_limits:
            msg = (
                f"Cannot call {func} as the space {self} has functional,"
                f" not rectangular limits. Use `rect_*` functions to obtain the"
                f" rectangular limits/area or `inside`/`filter` to test if values are"
                f" inside of the space."
            )
            raise RuntimeError(msg)

        return func(*args, **kwargs)

    return wrapped_func


@z.function(wraps="tensor", keepalive=True)
def calculate_rect_area(rect_limits):
    lower, upper = rect_limits
    diff = upper - lower
    return z.unstable.reduce_prod(diff, axis=-1)


@z.function(wraps="tensor", keepalive=True)
def inside_rect_limits(x, rect_limits):
    if (ndims := x.get_shape().ndims) is not None and ndims <= 1:
        msg = (
            "x has ndims <= 1, which is most probably not wanted. The default shape for array-like"
            " structures is (nevents, n_obs)."
        )
        raise ValueError(msg)
    lower, upper = z.unstack_x(rect_limits, axis=0)
    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)
    below_upper = znp.all(znp.less_equal(x, upper), axis=-1)  # if all obs inside
    above_lower = znp.all(znp.greater_equal(x, lower), axis=-1)
    return znp.logical_and(above_lower, below_upper)


@z.function(wraps="tensor", keepalive=True)
def filter_rect_limits(x, rect_limits, axis=None):
    return tf.boolean_mask(tensor=x, mask=inside_rect_limits(x, rect_limits=rect_limits, axis=axis))


def convert_to_tensor_or_numpy(obj, dtype=ztypes.float):
    if contains_tensor(obj):
        return znp.asarray(obj, dtype=dtype)
    else:
        with suppress(AttributeError):
            dtype = dtype.as_numpy_dtype
        return np.array(obj, dtype=dtype)


def _sanitize_x_input(x, n_obs):
    if isinstance(x, ZfitData):
        x = x.value()
    x = z.convert_to_tensor(x)
    if not x.shape.ndims > 1 and n_obs > 1:
        msg = (
            "x has ndims <= 1, which is most probably not wanted. The default shape for array-like"
            " structures is (nevents, n_obs)."
        )
        raise ValueError(msg)
    if x.shape.ndims <= 1 and n_obs == 1:
        x = tf.broadcast_to(x, (1, 1)) if x.shape.ndims == 0 else znp.expand_dims(x, axis=-1)
    if tf.get_static_value(x.shape[-1]) != n_obs:
        msg = (
            f"n_obs ({n_obs}) and the last dim of x (shape: {x.shape}) do not agree. Assuming x has shape (..., n_obs)"
        )
        raise ShapeIncompatibleError(msg)
    return x


def is_range_definition(limit):
    if isinstance(limit, (LimitRangeDefinition, ZfitSpace)):
        return True
    elif (isinstance(limit, np.ndarray) and limit.dtype != object) or tf.is_tensor(limit):
        return False
    try:
        return any(is_range_definition(lim) for lim in limit)
    except TypeError:
        return False  # not iterable and was not a LimitRangeDefinition in the beginning


# @tfp.experimental.auto_composite_tensor()
class Limit(
    ZfitLimit,
    # tfp.experimental.AutoCompositeTensor
):
    _experimental_allow_vectors = False

    def __init__(
        self,
        limit_fn: ztyping.LimitsFuncTypeInput = None,
        rect_limits: ztyping.LimitsTypeInput = None,
        n_obs: int | None = None,
    ):
        """Specify a limit with rectangular limits (and possiblty an arbitrary function).

        Args:
            limit_fn: Function that works as ``inside``: return true if a point is inside of the limits.
                The function should take one tensor-like argument with shape (..., n_obs) and should return
                a shape without the last dimension.
            rect_limits: Rectangular limits, a tuple of tensor-like objects with shape (typically) (1, n_obs) or similar
                such as only a tuple/list of values that will be interpreted as the last dimension. They should cover an
                area that includes ``limit_fn`` fully.
            n_obs: dimensionality of the Limits, the last dimension.
        """
        super().__init__()
        (
            limit_fn,
            rect_limits,
            n_obs,
            is_rect,
            sublimits,
        ) = self._check_convert_input_limits(limit_fn=limit_fn, rect_limits=rect_limits, n_obs=n_obs)
        self._limit_fn = limit_fn
        self._rect_limits = rect_limits
        self._n_obs = n_obs
        self._is_rect = is_rect
        self._sublimits = sublimits

    def _check_convert_input_limits(self, limit_fn, rect_limits, n_obs):
        if isinstance(limit_fn, ZfitLimit):
            if not isinstance(limit_fn, Limit):  # because of the limit_fn, that is private. Maybe use `inside` instead?
                msg = "If limits_fn is an instance of ZfitLimit, it has to be an instance of Limit (currently)"
                raise TypeError(msg)
            if rect_limits is not None or n_obs != limit_fn.n_obs:
                msg = "limits_fn is a ZfitLimit. rect_limits and n_obs must not be specified" "(or n_obs coincide)."
                raise OverdefinedError(msg)
            limit = limit_fn

            limit_fn = limit.limit_fn
            rect_limits = limit.rect_limits
            n_obs = limit.n_obs
        limits_are_rect = True

        # if the limits are False or None, we can take a shortcut and don't need to do any preprocessing
        return_limits_short = False
        if limit_fn is False:
            if rect_limits in (False, None):
                limits_short = False
                return_limits_short = True
        elif limit_fn is None:
            if rect_limits is False:
                limits_short = False
                return_limits_short = True
            elif rect_limits is None:
                limits_short = None
                return_limits_short = True
            else:  # start from limits are anything, rect is None
                limit_fn = rect_limits
                rect_limits = None
        if return_limits_short:
            sublimits = [type(self)(limit_fn=limits_short, n_obs=1) for _ in range(n_obs)] if n_obs > 1 else (self,)
            return limits_short, limits_short, n_obs, limits_short, sublimits

        if not callable(limit_fn):  # limits_fn is actually rect_limits
            rect_limits = limit_fn
            limit_fn = None

        else:
            limits_are_rect = False
            if rect_limits in (None, False):
                msg = "Limits given as a function need also rect_limits, cannot be None or False"
                raise ValueError(msg)
        try:
            lower, upper = rect_limits
        except TypeError as err:
            msg = "The outermost shape of `rect_limits` has to be 2 to represent (lower, upper)."
            raise TypeError(msg) from err

        lower = self._sanitize_rect_limit(lower)
        upper = self._sanitize_rect_limit(upper)

        # vectors means more than one n_events, in the first dim
        if not self._experimental_allow_vectors:
            lower_nevents = tf.get_static_value(lower.shape[0])
            upper_nevents = tf.get_static_value(upper.shape[0])
            if lower_nevents != 1 or upper_nevents != 1:
                msg = (
                    "Vectors (limits with n_events != 1) are not allowed. Experimental"
                    " flag (_experimental_allow_vectors) can be switched on if desired."
                    " This happened most likely due to the new Space limits layout:"
                    " To create multiple limits, use the addition operator of simple spaces."
                )
                raise LimitsIncompatibleError(msg)

        lower_nobs = tf.get_static_value(lower.shape[-1])

        if lower_nobs != (upper_nobs := tf.get_static_value(upper.shape[-1])):
            msg = f"Last dimension of lower ({lower_nobs}) and upper ({upper_nobs}) have to coincide."
            raise ShapeIncompatibleError(msg)
        if n_obs is not None and lower_nobs != n_obs:
            msg = f"Inferred last dimension ({lower_nobs}) does not coincide with " f"given n_obs ({n_obs})"
            raise ShapeIncompatibleError(msg)

        if not any(is_range_definition(limit) for limit in (lower, upper)):
            tf.assert_greater(
                upper,
                lower,
                message="All upper limits have to be larger than the lower limits and are"
                " given as (lower, upper). Maybe (upper, lower) was entered?",
            )

        n_obs = lower_nobs  # in case it was None
        rect_limits = (lower, upper)

        # It can be that there is a function that depends on multiple dimensions, e.g. if we have
        # a `limit_fn` and n_obs > 1. But if we have only rectangular limits, we can split them up
        # which allows later (the Space) to better combine and get subspaces

        # create sublimits to iterate if possible
        sublimits = []
        if limits_are_rect and n_obs > 1:
            for i in range(n_obs):
                low = z.unstable.gather(lower, (i,), axis=-1)
                up = z.unstable.gather(upper, (i,), axis=-1)
                sublimits.append(type(self)(rect_limits=(low, up), n_obs=1))
        else:
            sublimits.append(self)

        sublimits = tuple(sublimits)

        return limit_fn, rect_limits, n_obs, limits_are_rect, sublimits

    @staticmethod
    def _sanitize_rect_limit(limit) -> ztyping.RectLowerReturnType:
        """Sanitize the input limit and return if it is numerical or not.

        Args:
            limit:

        Returns:
        """
        dtype = object if is_range_definition(limit) else ztypes.float  # as the above ANY
        limit = convert_to_tensor_or_numpy(limit, dtype=dtype)
        if len(limit.shape) == 0:
            limit = z.unstable.broadcast_to(limit, shape=(1, 1))
        if len(limit.shape) == 1:
            limit = z.unstable.expand_dims(limit, axis=0)
        return limit

    @property
    def has_rect_limits(self) -> bool:
        """If the limits are rectangular."""
        return self.has_limits and self._is_rect

    @property
    def limits(self) -> ztyping.LimitsReturnType:
        """Return the limits as a tuple of the limit function and the rectangular limits.

        Returns:
            A tuple of the limit function and the rectangular limits.
        """
        return self.rect_limits

    @property  # todo: remove, legacy object
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        """Return the rectangular limits as ``np.ndarray/tf.Tensor`` if they are set and not false.

            The rectangular limits can be used for sampling. They do not in general represent the limits
            of the object as a functional limit can be set and to check if something is inside the limits,
            the method :py:meth:`~Limit.inside` should be used.

            In order to test if the limits are False or None, it is recommended to use the appropriate methods
            ``limits_are_false`` and ``limits_are_set``.

        Returns:
            The lower and upper limits.
        Raises:
            LimitsNotSpecifiedError: If there are not limits set or they are False.
        """
        if not self.has_limits:
            msg = "Limits are False or not set, cannot return the rectangular limits."
            raise LimitsNotSpecifiedError(msg)
        return self._rect_limits

    @property
    def _rect_limits_tf(self) -> ztyping.RectLimitsTFReturnType:
        rect_limits = self._rect_limits
        if rect_limits in (None, False):
            return rect_limits
        lower = z.convert_to_tensor(rect_limits[0])
        upper = z.convert_to_tensor(rect_limits[1])
        return z.convert_to_tensor((lower, upper))

    @property
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        """Return the rectangular limits as ``np.ndarray``. Raise error if not possible.

        Rectangular limits are returned as numpy arrays which can be useful when doing checks that do not
        need to be involved in the computation later on as they allow direct interaction with Python as
        compared to ``tf.Tensor`` inside a graph function.


        Returns:
            A tuple of two ``np.ndarray`` with shape (1, n_obs) typically. The last
                dimension is always ``n_obs``, the first can be vectorized. This allows unstacking
                with `z.unstack_x()` as can be done with data.

        Raises:
            CannotConvertToNumpyError: In case the conversion fails.
        """
        lower, upper = self._rect_limits

        lower = z.unstable._try_convert_numpy(lower)
        upper = z.unstable._try_convert_numpy(upper)
        return lower, upper

    @property
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        """The lower, rectangular limits, equivalent to `rect_limits[0] with shape (..., n_obs)`

        Returns:
            The lower, rectangular limits as `np.ndarray` or `tf.Tensor`
        """
        return self.rect_limits[0]

    @property
    def rect_upper(self) -> ztyping.RectUpperReturnType:
        """The upper, rectangular limits, equivalent to `rect_limits[1]` with shape (..., n_obs)

        Returns:
            The lower, rectangular limits as `np.ndarray` or `tf.Tensor`
        """
        return self.rect_limits[1]

    def rect_area(self) -> float | np.ndarray | znp.array:
        """Calculate the total rectangular area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        return calculate_rect_area(rect_limits=self._rect_limits_tf)

    def inside(self, x: ztyping.XTypeInput, guarantee_limits: bool = False) -> ztyping.XTypeReturnNoData:
        """Test if `x` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            Return a boolean tensor-like object with the same shape as the input `x` except of the
                last dimension removed.
        """
        x = _sanitize_x_input(x, n_obs=self.n_obs)
        if not self.has_limits:
            msg = "Cannot call `inside` without limits defined."
            raise LimitsNotSpecifiedError(msg)
        if guarantee_limits and self.has_rect_limits:
            return tf.broadcast_to(True, x.shape)
        else:
            return self._inside(x, guarantee_limits)

    def _inside(self, x, guarantee_limits):
        del guarantee_limits
        if self.has_rect_limits:
            return inside_rect_limits(x, rect_limits=self._rect_limits_tf)
        else:
            return self._limit_fn(x)

    def filter(
        self,
        x: ztyping.XTypeInput,
        guarantee_limits: bool = False,
        axis: int | None = None,
    ) -> ztyping.XTypeReturnNoData:
        """Filter `x` by removing the elements along `axis` that are not inside the limits.

        This is similar to `tf.boolean_mask`.

        Args:
            x: Values to be checked whether they are inside of the limits. If not, the corresonding element (in the
                specified `axis`) is removed. The shape is expected to have the last dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.
            axis: The axis to remove the elements from. Defaults to 0.

        Returns:
            Return an object with the same shape as `x` except that along `axis` elements have been
                removed.
        """

        if not self.has_limits:
            msg = "Cannot call `filter` without limits defined."
            raise LimitsNotSpecifiedError(msg)
        x = _sanitize_x_input(x, n_obs=self.n_obs)

        # shortcut, everything already inside
        if guarantee_limits and self.has_rect_limits:
            return x

        return self._filter(x, guarantee_limits, axis=axis)

    def _filter(self, x, guarantee_limits, axis):
        return tf.boolean_mask(tensor=x, mask=self.inside(x, guarantee_limits=guarantee_limits), axis=axis)

    @property
    def limit_fn(self):
        return self._limit_fn

    @property
    def rect_limits_are_tensors(self) -> bool:
        """Return True if the rectangular limits are tensors.

        If a limit with tensors is evaluated inside a graph context, comparison operations will fail.

        Returns:
            If the rectangular limits are tensors.
        """
        try:
            _ = self.rect_limits_np
        except CannotConvertToNumpyError:
            return True
        else:
            return False

    @property
    def limits_are_set(self) -> bool:
        """If the limits have never explicitly been set to a limit or to False.

        Returns:
        """
        return self._rect_limits is not None

    @property
    def limits_are_false(self) -> bool:
        """If the limits have been set to False, so the object on purpose does not contain limits.

        Returns:
        """
        return self._rect_limits is False

    @property
    def has_limits(self) -> bool:
        """If there are limits set and they are not false.

        Returns:
        """
        return not (self.limits_are_false or (not self.limits_are_set))

    @property
    def n_obs(self) -> int:
        """Dimensionality, the number of observables, of the limits. Equals to the last axis in rectangular limits.

        Returns:
            Dimensionality of the limits.
        """
        return self._n_obs

    @property
    def n_events(self) -> int | None:
        """

        Returns:
            Return the number of events, the dimension of the first shape. If this is > 1 or None,
                it's vectorized.
        """
        if not self.has_limits:
            return 1
        return self.rect_lower.shape[0]

    def equal(self, other: object, allow_graph: bool = True) -> znp.array:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.

        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            A ``znp.array`` with the result of the comparison.
         Raises:
             IllegalInGraphModeError: if `allow_graph`
        """
        if not isinstance(other, ZfitLimit):
            return np.array(False)
        return equal_limits(self, other, allow_graph=allow_graph)

    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Returns:

        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return NotImplemented
        return self.equal(other, allow_graph=False)

    def less_equal(self, other: object, allow_graph: bool = True) -> znp.array:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.


        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return np.array(False)
        return less_equal_limits(self, other, allow_graph=allow_graph)

    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return NotImplemented
        return self.less_equal(other, allow_graph=False)

    def get_sublimits(self) -> Iterable[ZfitLimit]:
        """Splits itself into multiple sublimits with smaller n_obs.

        If this is not possible, if the limits are not rectangular, just returns itself.

        Returns:
            The sublimits if it was able to split.
        """
        return self._sublimits

    def __hash__(self) -> int:
        objects = (
            self._limit_fn,
            self.n_obs,
        )  # not rect limits, not hashable and unprecise
        return hash(tuple(objects))

    def __repr__(self) -> str:
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.n_obs < 5 and not self.n_events > 1:
            limits = self.v1.limits
        else:
            limits = "rectangular"

        return f"<zfit {class_name} rect_limits={limits}, limit_fn={not self.has_rect_limits}>"


def rect_limits_are_any(limit: ZfitLimit) -> bool:
    """True if all limits in limit are ANY objects."""
    if limit.rect_limits_are_tensors:
        return False
    return bool(all(isinstance(ele, Any) for lim in limit.rect_limits_np for ele in lim.flatten()))


def less_equal_limits(limit1: Limit, limit2: Limit, allow_graph=True) -> znp.array:
    if rect_limits_are_any(limit1) or rect_limits_are_any(limit2):
        return np.array(True)

    try:
        lower1, upper1 = limit1.rect_limits_np
        lower2, upper2 = limit2.rect_limits_np
    except CannotConvertToNumpyError as error:
        if not allow_graph:
            msg = (
                "Cannot use equality in graph mode, e.g. inside a `tf.function` decorated "
                "function. To retrieve a symbolic Tensor, use `.equal(..., allow_graph=True)`"
            )
            raise IllegalInGraphModeError(msg) from error

        lower1, upper1 = limit1.rect_limits
        lower2, upper2 = limit2.rect_limits

    lower_le = z.unstable.reduce_all(z.unstable.less_equal(lower1, lower2), axis=-1)
    upper_le = z.unstable.reduce_all(z.unstable.less_equal(upper1, upper2), axis=-1)
    rect_limits_le = z.unstable.logical_and(lower_le, upper_le)
    # if both are functional, they have to coincide
    if not (limit1.has_rect_limits or limit2.has_rect_limits):
        funcs_equal = limit1.limit_fn == limit2.limit_fn

    # if one is functional, one is rect: the bigger one can be rect
    elif not limit1.has_rect_limits and limit2.has_rect_limits:
        funcs_equal = np.array(True)
    else:
        funcs_equal = limit1.limit_fn == limit2.limit_fn
    return z.unstable.logical_and(rect_limits_le, funcs_equal)


def equal_limits(limit1: Limit, limit2: Limit, allow_graph=True) -> bool:
    # if both are functional, we just need to compare their functions; the rect limits are "irrelevant"
    if not (limit1.has_rect_limits or limit2.has_rect_limits):
        return np.array(limit1.limit_fn == limit2.limit_fn)

    # if one is functional, one is rect: they are not the same
    elif limit1.has_rect_limits ^ limit2.has_rect_limits:
        return np.array(False)

    try:
        lower, upper = limit1.rect_limits_np
        lower_other, upper_other = limit2.rect_limits_np
    except CannotConvertToNumpyError as error:
        # todo: why does it fail without False?
        if not allow_graph:
            msg = (
                "Cannot use equality in graph mode, e.g. inside a `tf.function` decorated "
                "function. To retrieve a symbolic Tensor, use `.equal(..., allow_graph=True)`"
            )
            raise IllegalInGraphModeError(msg) from error

        lower, upper = limit1.rect_limits
        lower_other, upper_other = limit2.rect_limits

    # TODO add tols
    lower_limits_equal = z.unstable.reduce_all(z.unstable.allclose_anyaware(lower, lower_other))
    upper_limits_equal = z.unstable.reduce_all(z.unstable.allclose_anyaware(upper, upper_other))
    rect_limits_equal = z.unstable.logical_and(lower_limits_equal, upper_limits_equal)
    funcs_equal = limit1.limit_fn == limit2.limit_fn
    return z.unstable.logical_and(rect_limits_equal, funcs_equal)


def deprecate_multispace(func):
    msg = (
        "The functionality of using a MultiSpace, that is, a space with multiple limits,"
        " has been deprecated in favor of using the `TruncatedPDF`, or generally, a list of spaces, if it is supported."
        "The `MultiSpace` will be removed in the future, as it is too complicated with too little usage."
        "If you think a use-case is missing or have any questions, please open an issue on GitHub: https://github.com/zfit/zfit/issues/new"
    )

    return deprecated(None, msg)(func)


warned_new_limits = False


def warn_new_limits(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global warned_new_limits
        if not warned_new_limits:
            msg = (
                "The attribute `limits`, `lower` and `upper` will have a changed behavior with the future 1.0 release."
                "They will have one dimension less, making them more intuitive: `limits` will return a tuple of one dimensional arrays,"
                " each dimension corresponding to one observable. `lower` and `upper` will return a one dimensional array for each observable."
                "Furthermore"
                "To stay compatible and not experience breaking changes, use `Space.v1.limits`, `Space.v1.lower` and `Space.v1.upper` for the future behavior, "
                "or use `Space.v0.limits`, `Space.v0.lower` and `Space.v0.upper` to keep the current behavior (discouraged)."
            )
            warnings.warn(msg, FutureWarning, stacklevel=2)
            warned_new_limits = True
        return func(*args, **kwargs)

    return wrapper


class BaseSpace(ZfitSpace, BaseObject):
    def __init__(self, obs, axes, name, **kwargs):
        super().__init__(name, **kwargs)
        coords = Coordinates(obs, axes)
        self.coords = coords

    @property
    def is_binned(self):
        return self.binning is not None

    def inside(self, x: ztyping.XTypeInput, guarantee_limits: bool = False) -> ztyping.XTypeReturn:
        """Test if `x` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            Return a boolean tensor-like object with the same shape as the input `x` except of the
                last dimension removed.
        """
        x = _sanitize_x_input(x, n_obs=self.n_obs)
        if self.has_rect_limits and guarantee_limits:
            return tf.broadcast_to(True, x.shape)
        return self._inside(x, guarantee_limits)

    @deprecated(
        None, "Use the `volume` attribute instead (don't call it, i.e. convert `space.area()` to `space.volume`."
    )
    def area(self) -> float | znp.array:
        return self._legacy_area()

    @property
    def volume(self) -> float | znp.array:
        return self._legacy_area()

    @abstractmethod
    def _inside(self, x, guarantee_limits):
        raise NotImplementedError

    def filter(
        self,
        x: ztyping.XTypeInput,
        guarantee_limits: bool = False,
        axis: int | None = None,
    ) -> ndarray | Tensor | DataFrame | Any:
        """Filter `x` by removing the elements along `axis` that are not inside the limits.

        This is similar to `tf.boolean_mask`.

        Args:
            x: Values to be checked whether they are inside of the limits. If not, the corresonding element (in the
                specified `axis`) is removed. The shape is expected to have the last dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.
            axis: The axis to remove the elements from. Defaults to 0.

        Returns:
            Return an object with the same shape as `x` except that along `axis` elements have been
                removed.
        """
        if axis is not None:
            msg = "Axis is not yet implemented."
            raise ValueError(msg)
        if self.has_rect_limits and guarantee_limits:
            return x
        if isinstance(x, pd.DataFrame):
            exprs = []
            if missing_obs := set(self.obs) - set(x.columns):  # todo: expose the expression?
                msg = f"Dataframe is missing the following observables: {missing_obs}"
                raise ValueError(msg)
            for ob, lower, upper in zip(self.obs, self.v1.lower, self.v1.upper):
                expr = f"({lower} <= {ob} <= {upper})"
                exprs.append(expr)
            expr = " & ".join(exprs)
            return x.query(expr)
        return self._filter(x, guarantee_limits)

    def _filter(self, x, guarantee_limits):
        if isinstance(x, ZfitData):
            x = x.value()
        return tf.boolean_mask(tensor=x, mask=self.inside(x, guarantee_limits=guarantee_limits))

    @property
    def n_obs(self) -> int:
        """Return the number of observables/axes.

        Returns:
        Returns:
            int >= 1
        """
        return self.coords.n_obs

    @property
    def obs(self) -> ztyping.ObsTypeReturn:
        """The observables ("axes with str")the space is defined in.

        Returns:
        """
        return self.coords.obs

    @property
    def axes(self) -> ztyping.AxesTypeReturn:
        """The axes ("obs with int") the space is defined in.

        Returns:
        """
        return self.coords.axes

    @property
    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement: https://github.com/zfit/zfit/discussions/533",
    )
    def n_limits(self) -> int:
        return self._depr_n_limits()

    @property
    def _depr_n_limits(self):
        return len(tuple(self))

    def __iter__(self) -> Iterable[ZfitSpace]:
        yield self

    def get_reorder_indices(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None) -> tuple[int]:
        """Indices that would order the instances obs as `obs` respectively the instances axes as `axes`.

        Args:
            obs: Observables that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.
            axes: Axes that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.

        Returns:
            New indices that would reorder the instances obs to be obs respectively axes.

        Raises:
            CoordinatesUnderdefinedError: If neither `obs` nor `axes` is given
        """
        return self.coords.get_reorder_indices(obs=obs, axes=axes)

    # TODO: remove, in coords
    def _check_convert_input_axes(
        self, axes: ztyping.AxesTypeInput, allow_none: bool = False
    ) -> ztyping.AxesTypeReturn:
        if axes is None:
            if allow_none:
                return None
            else:
                msg = "TODO: Cannot be None"
                raise AxesNotSpecifiedError(msg)
        # TODO(Mayou36): extend like _check_obs?
        return axes.axes if isinstance(axes, ZfitSpace) else convert_to_container(value=axes, container=tuple)

    # TODO: remove, in coords
    def _check_convert_input_obs(self, obs: ztyping.ObsTypeInput, allow_none: bool = False) -> ztyping.ObsTypeReturn:
        """Input check: Convert `NOT_SPECIFIED` to None or check if obs are all strings.

        Args:
            obs:

        Returns:
        """
        if obs is None:
            if allow_none:
                return None
            else:
                msg = "TODO: Cannot be None"
                raise ObsNotSpecifiedError(msg)

        if isinstance(obs, ZfitSpace):
            obs = obs.obs
        else:
            obs = convert_to_container(obs, container=tuple)
            obs_not_str = tuple(o for o in obs if not isinstance(o, str))
            if obs_not_str:
                msg = f"The following observables are not strings: {obs_not_str}"
                raise ValueError(msg)
        return obs

    def _check_coords_allowed(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ):
        to_check = []
        if obs is not None and self.obs is not None:
            to_check.append(obs, self.obs)
        if axes is not None and self.axes is not None:
            to_check.append(axes, self.axes)

        for coord, self_coord in to_check:
            coord = frozenset(coord)
            self_coord = frozenset(self_coord)
            if coord != self_coord:
                if not allow_superset and coord.issuperset(self_coord):
                    msg = f"Superset is not allowed, but {coord} is a superset" f" of {self_coord}"
                    raise CoordinatesIncompatibleError(msg)

                if not allow_subset and coord.issubset(self_coord):
                    msg = f"subset is not allowed, but {coord} is a subset" f" of {self_coord}"
                    raise CoordinatesIncompatibleError(msg)

    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.has_rect_limits:
            limits = self.rect_limits if self.n_obs < 3 and not self.n_events > 1 else "rectangular"
        else:
            limits = "functional"
        return f"<zfit {class_name} obs={self.obs}, axes={self.axes}, limits={limits}, binned={self.is_binned}>"

    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement: https://github.com/zfit/zfit/discussions/533",
    )
    def __add__(self, other):
        if not isinstance(other, ZfitSpace):
            msg = f"Cannot add a {type(self)} and a {type(other)}"
            raise TypeError(msg)
        return add_spaces(self, other)

    def get_sublimits(self):
        limits = self.extract_limits()
        return list(limits.values())

    @deprecate_multispace
    def add(self, *other: ztyping.SpaceOrSpacesTypeInput):
        """Add the limits of the spaces. Only works for the same obs.

        In case the observables are different, the order of the first space is taken.

        Args:
            other:

        Returns:
            :py:class:`~zfit.Space`:
        """
        # other = convert_to_container(other, container=list)
        return add_spaces(self, *other)

    def combine(self, *other: ztyping.SpaceOrSpacesTypeInput) -> ZfitSpace:
        """Combine spaces with different obs (but consistent limits).

        Args:
            other:

        Returns:
            :py:class:`~zfit.Space`:
        """
        # other = convert_to_container(other, container=list)
        return combine_spaces(self, *other)

    def __mul__(self, other):
        return self.combine(other)

    def __ge__(self, other):
        return NotImplemented

    def equal(self, other: object, allow_graph: bool) -> znp.array:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return False
        return equal_space(self, other, allow_graph=allow_graph)

    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Returns:

        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return NotImplemented
        return self.equal(other=other, allow_graph=False)

    def less_equal(self, other, allow_graph):
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return False
        return less_equal_space(self, other, allow_graph=allow_graph)

    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.less_equal(other, allow_graph=False)

    def __hash__(self):
        limits_frozen = tuple(((key, tuple(ldict.items())) for key, ldict in self._limits_dict.items()))
        return hash((limits_frozen, hash(self.coords), hash(self.binning)))

    def reorder_x(self, x, x_obs, x_axes, func_obs, func_axes):
        return self.coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes, func_obs=func_obs, func_axes=func_axes)

    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement: https://github.com/zfit/zfit/discussions/533",
    )
    def __len__(self):
        if not self:
            return 0
        else:
            return sum(1 for _ in self)

    def __bool__(self):
        return self.has_limits


# @tfp.experimental.auto_composite_tensor()
def _legacy_get_arguments_space(obs, args, limits, binning, axes, rect_limits, lower, upper):  # noqa: ARG001
    """Legacy function to get the arguments of a `Space`, i.e. making the transition to allow "lower" and "upper"
    smooth."""

    if len(args) > 2:  # catch legacy behavior
        msg = "The API for `Space` has changed and takes now lower, upper as the first two arguments. Cannot take more than 3 positional arguments"
        raise BreakingAPIChangeError(msg)

    if len(args) == 2:
        if all(isinstance(lim, (int, float, Any)) for lim in args):
            limits = args
        else:
            limits, binning = args
    elif len(args) == 1:
        limits = args[0]
    elif lower is not None and upper is not None:
        limits = [lower, upper]
        # elif lower is None and upper is None and limits is None and rect_limits is None:
        #     raise BreakingAPIChangeError(
        #         "The API for `Space` has changed and takes now lower, upper. Could not deduce the intention, as only one of lower, upper was given")

    return limits, binning


class Space(
    BaseSpace,
    SerializableMixin,
    # tfp.experimental.AutoCompositeTensor
):
    AUTO_FILL = object()
    ANY = ANY
    ANY_LOWER = ANY_LOWER  # TODO: needed? or move everything inside?
    ANY_UPPER = ANY_UPPER

    @deprecated_args(None, "Use `lower` and `upper` instead", ("rect_limits", "limits"))
    def __init__(
        self,
        obs: ztyping.ObsTypeInput | None = None,
        *args,
        limits: ztyping.LimitsTypeInput | None = None,
        binning: ztyping.BinningTypeInput = None,
        axes=None,
        rect_limits=None,
        name: str | None = None,
        label: Union[str, Iterable[str]] | None = None,
        lower: ztyping.LimitsTypeInputV1 | None = None,
        upper: ztyping.LimitsTypeInputV1 | None = None,
    ):
        """Define a space with the name (`obs`) of the axes (and it's number) and possibly it's limits.

        A space can be thought of as coordinates, possibly with the definition of a range (limits). For most use-cases,
        it is sufficient to specify a `Space` via observables; simple string identifiers. They can be multidimensional.

        Observables are like the columns of a spreadsheet/dataframe, and are therefore needed for any object that does
        numerical operations or holds data in order to match the right axes. On object creation, the observables are
        assigned using a `Space`. This is often used as the default space of an object and can be used as the
        default `norm`, sampling limits etc.

        Axes are the same concept as observables, but numbers, indexes, and are used *inside* an object. There,
        axes 0 corresponds to the 0th data column we get (which corresponds to a certain observable).

        Args:
            obs: |@doc:space.init.obs| Observable of the space.
               Serves as the "variable". |@docend:space.init.obs|
            lower, upper: |@doc:space.init.lowerupper| Lower and upper limits of the space, respectively.
               Each of them should be a scalar-like object. |@docend:space.init.lowerupper|
            limits: |@doc:space.init.limits| A tuple-like object of the limits of the space.
               These are the lower and upper limits. |@docend:space.init.limits|
            binning: |@doc:space.init.binning| Binning of the space.
               Currently, only regular and variable binning *with a name* is supported.
               If an integer or a list of integers is given with
               lengths equal to the number of observables,
               it is interpreted as the number of bins and
               a regular binning is automatically created using the limits as the
               start and end points. |@docend:space.init.binning|
            name: |@doc:space.init.name| Name of the space.
               Maybe has implications on the serialization and deserialization of the space.
               For a human-readable name, use the label. |@docend:space.init.name|

        Raises
            TypeError: If the axes in the binning do not have a name.
            ObsIncompatibleError: If the obs do not agree with the name of the binning.
            ShapeIncompatibleError: If the shape of the limits or the binnings do not match the shape of the obs.
        """

        self.v1 = V1Space(self)
        self.v0 = V0Space(self)
        self.vec = VectorizeLimits(self)

        limits, binning = _legacy_get_arguments_space(obs, args, limits, binning, axes, rect_limits, lower, upper)
        # limits = [lower, upper]  # temporary
        if binning is None:
            binning = False
        if name is None:
            name = "Space"
        if binning is not False:
            integer_autobinning = isinstance(binning, int) or (
                isinstance(binning, (list, tuple)) and all(isinstance(b, int) for b in binning)
            )
            if not integer_autobinning:
                if not isinstance(binning, Binnings):
                    binning = convert_to_container(binning)
                    if binning is not None:
                        binning = histaxes_to_binning(binning)
                if not all(binning.name):
                    msg = f"Axes must have a name. Missing: {[axis for axis in binning if not hasattr(axis, 'name')]}"
                    raise TypeError(msg)
                if obs is None and axes is None:
                    obs = [axis.name for axis in binning]
        super().__init__(obs=obs, axes=axes, name=name)

        label = convert_to_container(label, container=tuple)
        if label is not None and len(label) != self.n_obs:
            msg = f"Number of labels ({len(label)}) does not match the number of observables ({self.n_obs})"
            raise ValueError(msg)

        if self.obs is not None:  # if None, the obs are not given, only axes -> no labels
            if label is None:
                label = self.obs
            elif len(label) != self.n_obs:
                msg = f"Number of labels ({label}) does not match the number of observables ({self.obs})"
                raise ValueError(msg)
            label = dict(zip(self.obs, label))
        self._labels = label

        if binning is not False and not isinstance(binning, int) and limits is None and rect_limits is None:
            limits = [[], []]
            for axis in binning:
                limits[0].append(axis.edges[0])
                limits[1].append(axis.edges[-1])

        limits_dict = self._check_convert_input_limits(
            limit=limits,
            rect_limits=rect_limits,
            obs=self.obs,
            axes=self.axes,
            n_obs=self.n_obs,
        )
        self._limits_dict = limits_dict

        if binning is not False and isinstance(binning, int):
            binning = [binning]
        if binning is not False and integer_autobinning:
            if len(binning) != self.n_obs:
                msg = (
                    f"Wrong number ({len(binning)}) of integers given for regular binning"
                    f" ({binning}) with {self.n_obs} observables ({self.obs})."
                    f" Numbers have to match the number of observables."
                )
                raise ShapeIncompatibleError(msg)
            regular_binnings = []
            for i, nbins in enumerate(binning):
                if nbins < 1:
                    msg = "If binning is an integer, it must be > 0"
                    raise ValueError(msg)

                lower = self.lower[0][i]
                upper = self.upper[0][i]
                regular_binnings.append(RegularBinning(bins=nbins, start=lower, stop=upper, name=self.obs[i]))

            binning = Binnings(regular_binnings)
        if binning is not False:
            bining_names = set(binning.name)
            obs = set(self.obs)
            wrong_names = bining_names - obs
            if wrong_names:
                msg = f"Binning names ({wrong_names}) do not match observables ({obs}), {wrong_names} not in space."
                raise ObsIncompatibleError(msg)
            missing_obs = obs - bining_names
            if missing_obs:
                msg = f"Binning names ({missing_obs}) do not match observables ({obs}), missing {missing_obs}."
                raise ObsIncompatibleError(msg)
            binning = Binnings([binning[ob] for ob in self.obs])
        self._binning = None if binning is False else binning

    @property
    def labels(self):
        if (obs := self.obs) is not None:
            return tuple(self._labels[ob] for ob in obs)
        return None  # we have axis -> no labels

    @property
    def label(self):
        if self.n_obs > 1:
            msg = f"{self} has more than one observable, use `labels` instead."
            raise ValueError(msg)
        return self.labels[0]

    @property
    def binning(self):
        return self._binning
        # if binning_out is not None:
        #     binning_out =

    @property
    def is_binned(self):
        return self.binning is not None

    def _check_convert_input_limits(
        self,
        limit: ztyping.LowerTypeInput | ztyping.UpperTypeInput,
        rect_limits,
        obs,
        axes,
        n_obs,
    ) -> ztyping.LowerTypeReturn | ztyping.UpperTypeReturn:
        """Check and sanitize the input limits as well as the rectangular limits.

        Args:
            limit:

        Returns:
            Limits dictionary containing the observables and/or the axes as a key matching
                `ZfitLimits` objects.
        """
        limits_dict = defaultdict(dict)
        input_limits = limit
        if isinstance(input_limits, Space):
            space = input_limits

            # get the subset of obs/axes, then drop the other coord if not given
            if obs and not axes:
                space = space.with_obs(obs, allow_subset=True, allow_superset=True)
                space = space.with_axes(None)
            elif not obs and axes:
                space = space.with_axes(axes, allow_subset=True, allow_superset=True)
                space = space.with_obs(None)
            elif obs and axes:
                coords = Coordinates(obs=obs, axes=axes)
                space = space.with_coords(coords, allow_superset=True, allow_subset=True)
            input_limits = space.get_limits()

            obs = space.obs
            axes = space.axes
        if not isinstance(input_limits, dict) and not isinstance(rect_limits, dict):
            # if not input_limits and rect_limits:
            #     input_limits = rect_limits
            limit = Limit(limit_fn=limit, rect_limits=rect_limits, n_obs=n_obs)
            i_old = 0
            for lim in limit.get_sublimits():  # split into smaller ones if possible
                i = i_old + lim.n_obs
                if obs is not None:
                    limits_dict["obs"][obs[i_old:i]] = lim
                if axes is not None:
                    limits_dict["axes"][axes[i_old:i]] = lim
                i_old = i
            input_limits = limits_dict

        if isinstance(input_limits, dict):
            input_limits = input_limits.copy()
        elif isinstance(rect_limits, dict):
            input_limits = rect_limits.copy()

        if "axes" not in input_limits and "obs" not in input_limits:
            msg = "Probably internal error: wrong format of limits_dict"
            raise ValueError(msg)

        # check if obs is in the limits dict. If not, copy it from the axes
        if obs:
            if "obs" in input_limits:
                obs_limit_dict = input_limits["obs"]
                obs_limit_dict = {ob: lim for ob, lim in obs_limit_dict.items() if ob[0] in obs}
            else:
                obs_limit_dict = {}
                for axes_lim, lim in input_limits["axes"].items():
                    obs_coords = tuple(obs[axes.index(ax)] for ax in axes_lim)
                    if isinstance(lim, ZfitOrderableDimensional):
                        lim = lim.with_coords(self.space)
                    obs_limit_dict[obs_coords] = lim
            limits_dict["obs"] = obs_limit_dict

        if axes:
            if "axes" in input_limits:
                axes_limit_dict = input_limits["axes"]
                axes_limit_dict = {axis: lim for axis, lim in axes_limit_dict.items() if axis[0] in axes}
            else:
                axes_limit_dict = {}
                for obs_lim, lim in input_limits["obs"].items():
                    axes_coords = tuple(axes[obs.index(ob)] for ob in obs_lim)

                    if isinstance(lim, ZfitOrderableDimensional):
                        lim = lim.with_coords(self.space)
                    axes_limit_dict[axes_coords] = lim
            limits_dict["axes"] = axes_limit_dict

        if not axes and "axes" in limits_dict:
            limits_dict.pop("axes")

        if not obs and "obs" in limits_dict:
            limits_dict.pop("obs")

        return limits_dict

    def get_limits(
        self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None
    ) -> ztyping.LimitsDictWithCoords | ztyping.LimitsDictNoCoords:
        return_dict = {}
        both_none_or_true = obs is None and axes is None or obs is True and axes is True

        if obs is True and axes is None or both_none_or_true:
            if not self.obs:
                if obs is True:
                    msg = "Obs are not defined for this instance, no limits set for obs."
                    raise ObsIncompatibleError(msg)
            else:
                return_dict["obs"] = self._limits_dict["obs"].copy()
        if axes is True and obs is None or both_none_or_true:
            if not self.axes:
                if axes is True:
                    msg = "Axes are not defined for this instance, no limits set for axes."
                    raise AxesIncompatibleError(msg)
            else:
                return_dict["axes"] = self._limits_dict["axes"].copy()
        else:
            if obs:
                return_dict["obs"] = extract_limits_from_dict(self._limits_dict, obs=obs)
            if axes:
                return_dict["axes"] = extract_limits_from_dict(self._limits_dict, axes=axes)
        return return_dict

    @property
    @fail_not_rect
    def limits(self) -> ztyping.LimitsTypeReturn:
        """Return the limits.

        Returns:
        """
        return self.rect_limits

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        """Return the rectangular limits as `np.ndarray``tf.Tensor` if they are set and not false.

            The rectangular limits can be used for sampling. They do not in general represent the limits
            of the object as a functional limit can be set and to check if something is inside the limits,
            the method :py:meth:`~Limit.inside` should be used.

            In order to test if the limits are False or None, it is recommended to use the appropriate methods
            `limits_are_false` and `limits_are_set`.

        Returns:
            The lower and upper limits.
        Raises:
            LimitsNotSpecifiedError: If there are not limits set or they are False.
        """
        return self._depr_rect_limits

    @property
    def _depr_rect_limits(self):
        if not self.has_limits:
            msg = "Limits are False or not set, cannot return the rectangular limits."
            raise LimitsNotSpecifiedError(msg)
        lower_ordered, upper_ordered = self._rect_limits_z()
        return lower_ordered, upper_ordered

    @property
    def _rect_limits_tf(self) -> ztyping.LimitsTypeReturn:
        """Return the limits as `tf.Tensor`.

        Returns:
        """
        if not self.has_limits:
            msg = "Limits are False or not set, cannot return the rectangular limits."
            raise LimitsNotSpecifiedError(msg)
        lower_ordered, upper_ordered = self._rect_limits_z()
        return znp.asarray(lower_ordered), znp.asarray(upper_ordered)

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        """Return the rectangular limits as `np.ndarray`. Raise error if not possible.

        Rectangular limits are returned as numpy arrays which can be useful when doing checks that do not
        need to be involved in the computation later on as they allow direct interaction with Python as
        compared to `tf.Tensor` inside a graph function.

        In order to test if the limits are False or None, it is recommended to use the appropriate methods
        `limits_are_false` and `limits_are_set`.

        Returns:
            A tuple of two `np.ndarray` with shape (1, n_obs) typically. The last
                dimension is always `n_obs`, the first can be vectorized. This allows unstacking
                with `z.unstack_x()` as can be done with data.

        Raises:
            CannotConvertToNumpyError: In case the conversion fails.
            LimitsNotSpecifiedError: If the limits are not set
        """
        lower, upper = self._rect_limits_z()

        lower = z.unstable._try_convert_numpy(lower)
        upper = z.unstable._try_convert_numpy(upper)
        return lower, upper

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_upper(self) -> ztyping.UpperTypeReturn:
        """The upper, rectangular limits, equivalent to `rect_limits[1]` with shape (..., n_obs)

        Returns:
            The upper, rectangular limits as `np.ndarray` or `tf.Tensor`
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        return self.v0.limits[1]

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        """The lower, rectangular limits, equivalent to `rect_limits[0]` with shape (..., n_obs)

        Returns:
            The lower, rectangular limits as `np.ndarray` or `tf.Tensor`
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        return self.v0.limits[0]

    def _rect_limits_z(self):
        limits_coords = []
        rect_lower_unordered = []
        rect_upper_unordered = []
        obs_in_use = self.obs is not None
        limits_dict = self._limits_dict["obs" if obs_in_use else "axes"]

        for (
            coord_limit,
            limit,
        ) in limits_dict.items():  # TODO: maybe refactor with extract limits?
            limits_coords.extend(coord_limit)
            lower, upper = limit.rect_limits  # to get the numpy or tensor
            rect_lower_unordered.append(lower)
            rect_upper_unordered.append(upper)
        reorder_kwargs = {"x_obs" if obs_in_use else "x_axes": limits_coords}

        # stack the limits and reorder them according to the own coords
        lower_stacked = z.unstable.concat(rect_lower_unordered, axis=-1)
        lower_ordered = self.reorder_x(lower_stacked, **reorder_kwargs)
        upper_stacked = z.unstable.concat(rect_upper_unordered, axis=-1)
        upper_ordered = self.reorder_x(upper_stacked, **reorder_kwargs)
        return lower_ordered, upper_ordered

    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_area(self) -> float | np.ndarray | znp.array:
        """Calculate the total rectangular area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        return self._legacy_area()

    @property
    def rect_limits_are_tensors(self) -> bool:
        """Return True if the rectangular limits are tensors.

        If a limit with tensors is evaluated inside a graph context, comparison operations will fail.

        Returns:
            If the rectangular limits are tensors.
        """
        try:
            _ = self.rect_limits_np
        except CannotConvertToNumpyError:
            return True
        else:
            return False

    @property
    def has_rect_limits(self) -> bool:
        """If there are limits and whether they are rectangular."""
        limits_dict = self._limits_dict.get("obs") if self.obs is not None else self._limits_dict.get("axes")
        if not limits_dict:
            return False
        rect_limits = [limit.has_rect_limits for limit in limits_dict.values()]
        all_rect_limits = all(rect_limits)
        return all_rect_limits and len(rect_limits) > 0

    @property
    def limits_are_false(self) -> bool:
        """If the limits have been set to False, so the object on purpose does not contain limits.

        Returns:
            True if limits is False
        """
        return all(limit.limits_are_false for limit in self._limits_dict["obs" if self.obs else "axes"].values())

    @property
    def has_limits(self) -> bool:
        """Whether there are limits set and they are not false.

        Returns:
        """
        return self.limits_are_set and not self.limits_are_false

    @property
    def limits_are_set(self):
        return all(
            limit.limits_are_set
            for limit in self._limits_dict["obs" if self.obs else "axes"].values()
            if limit is not self
        )

    @property
    def n_events(self) -> int | None:
        """Return the number of events, the dimension of the first shape.

        Returns:
            Number of events, the dimension of the first shape. If this is > 1 or None,
                it's vectorized.
        """
        if not self.has_limits:
            return 1
        return self.rect_lower.shape[0]

    @property
    @fail_not_rect
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits.

        Returns:
        """
        return self.rect_lower

    @property
    @fail_not_rect
    def upper(self) -> ztyping.UpperTypeReturn:
        """Return the upper limits.

        Returns:
        """
        return self.rect_upper

    @property
    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement:"
        "https://github.com/zfit/zfit/discussions/533",
    )
    def n_limits(self) -> int:
        """The number of different limits.

        Returns:
            int >= 1
        """
        return len(tuple(self))

    # def with_limits(
    #         self,
    #         *args,
    #         limits: ztyping.LimitsTypeInput = None,
    #         rect_limits: ztyping.RectLimitsInputType | None = None,
    #         lower: ztyping.LimitsTypeInputV1 | None = None,
    #         upper: ztyping.LimitsTypeInputV1 | None = None,
    #         name: str | None = None,
    # ) -> ZfitSpace:
    #     """Return a copy of the space with the new `limits` (and the new `name`).
    #
    #     Args:
    #         limits: Limits to use. Can be rectangular, a function (requires to also specify `rect_limits`
    #             or an instance of ZfitLimit.
    #         rect_limits: Rectangular limits that will be assigned with the instance
    #         name: Human readable name
    #
    #     Returns:
    #         Copy of the current object with the new limits.
    #     """
    #     if len(args) > 2:
    #         msg = "Cannot take more than 3 positional arguments"
    #         raise ValueError(msg)
    #     if len(args) == 2:
    #         limits, rect_limits = args
    #     elif len(args) == 1:
    #         limits = args[0]
    #     if limits is not None and rect_limits is not None and all(isinstance(lim, (int, float, Any)) for lim in (limits, rect_limits)):
    #         if lower is not None or upper is not None:
    #             msg = "Cannot set both limits and lower, upper."
    #             raise ValueError(msg)
    #         lower, upper = limits, rect_limits
    #
    #     limits = [lower, upper]
    #     rect_limits = limits
    #
    #     return type(self)(
    #         obs=self.coords,
    #         limits=limits,
    #         rect_limits=rect_limits,
    #         binning=self.binning,
    #         name=name,
    #     )

    def with_limits(
        self,
        limits: ztyping.LimitsTypeInput = None,
        rect_limits: ztyping.RectLimitsInputType | None = None,
        name: str | None = None,
    ) -> ZfitSpace:
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits: Limits to use. Can be rectangular, a function (requires to also specify `rect_limits`
                or an instance of ZfitLimit.
            rect_limits: Rectangular limits that will be assigned with the instance
            name: Human readable name

        Returns:
            Copy of the current object with the new limits.
        """
        return type(self)(
            obs=self.coords, limits=limits, rect_limits=rect_limits, binning=self.binning, name=name, label=self.labels
        )

    def reorder_x(
        self,
        x: tf.Tensor | np.ndarray,
        *,
        x_obs: ztyping.ObsTypeInput = None,
        x_axes: ztyping.AxesTypeInput = None,
        func_obs: ztyping.ObsTypeInput = None,
        func_axes: ztyping.AxesTypeInput = None,
    ) -> ztyping.XTypeReturnNoData:
        """Reorder x in the last dimension either according to its own obs or assuming a function ordered with func_obs.

        There are two obs or axes around: the one associated with this Coordinate object and the one associated with x.
        If x_obs or x_axes is given, then this is assumed to be the obs resp. the axes of x and x will be reordered
        according to `self.obs` resp. `self.axes`.

        If func_obs resp. func_axes is given, then x is assumed to have `self.obs` resp. `self.axes` and will be
        reordered to align with a function ordered with `func_obs` resp. `func_axes`.

        Switching `func_obs` for `x_obs` resp. `func_axes` for `x_axes` inverts the reordering of x.

        Args:
            x: Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, `func_obs` and `func_axes` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:
            The reordered array-like object
        """
        return self.coords.reorder_x(x=x, x_obs=x_obs, x_axes=x_axes, func_obs=func_obs, func_axes=func_axes)

    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitSpace:
        """Create a new Space that has `obs`; sorted by or set or dropped.

        The behavior is as follows:

         * obs are already set:

           * input obs are None: the observables will be dropped. If no axes are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input obs are not None: the instance will be sorted by the incoming obs. If axes or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned.
             This can be used to take a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the obs will be sorted accordingly as
             if the obs not contained in the instances obs were not in the input obs.
         * obs are not set:

           * if the input obs are None, the same object is returned.
           * if the input obs are not None, they will be set as-is and now correspond to the already
             existing axes in the object.

        Args:
            obs: Observables to sort/associate this instance with
            allow_superset: if False and a strict superset of the own observables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own observables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/observables

        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            ObsIncompatibleError: if `obs` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if obs is None:  # drop obs, check if there are axes
            if self.obs is None:
                return self
            if self.axes is None:
                msg = "Cannot remove obs (using None) for a Space without axes"
                raise AxesIncompatibleError(msg)
            new_limits = self._limits_dict.copy()
            new_space = self.copy(obs=obs, limits=new_limits)
        else:
            obs = _convert_obs_to_str(obs)
            coords = self.coords.with_obs(obs, allow_superset=allow_superset, allow_subset=allow_subset)
            binning = self.binning
            if binning is not None:
                binning = [binning[ob] for ob in obs if ob in self.obs]
            if (newlabels := self.labels) is not None:
                newlabels = tuple(self._labels[ob] for ob in coords.obs)  # use just the obs that are available
            new_space = type(self)(coords, limits=self._limits_dict, binning=binning, label=newlabels)
        return new_space

    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitSpace:
        """Create a new instance that has `axes`; sorted by or set or dropped.

        The behavior is as follows:

         * axes are already set:

           * input axes are None: the axes will be dropped. If no observables are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input axes are not None: the instance will be sorted by the incoming axes. If obs or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned. This can
             be used to retrieve a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the axes will be sorted accordingly as
             if the axes not contained in the instances axes were not present in the input axes.
         * axes are not set:

           * if the input axes are None, the same object is returned.
           * if the input axes are not None, they will be set as-is and now correspond to the already
             existing obs in the object.

        Args:
            axes: Axes to sort/associate this instance with
            allow_superset: if False and a strict superset of the own axeservables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own axeservables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/axes
        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            AxesIncompatibleError: if `axes` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if axes is None:  # drop axes
            if self.axes is None:
                return self
            if self.obs is None:
                msg = "Cannot remove axes (using None) for a Space without obs"
                raise ObsIncompatibleError(msg)
            new_limits = self._limits_dict.copy()
            if (newlabels := self.labels) is not None and (obs := self.obs) is not None:
                newlabels = tuple(self._labels[ob] for ob in obs)
            new_space = self.copy(axes=axes, limits=new_limits, label=newlabels)
        else:
            axes = convert_to_axes(axes)
            if self.axes is None:
                if len(axes) != len(self.obs):
                    msg = f"Trying to set axes {axes} to object with obs {self.obs}"
                    raise AxesIncompatibleError(msg)
                if (newlabels := self.labels) is not None and (obs := self.obs) is not None:
                    newlabels = tuple(self._labels[ob] for ob in obs)
                new_space = self.copy(axes=axes, limits=self._limits_dict, label=newlabels)
            else:
                coords = self.coords.with_axes(axes=axes, allow_superset=allow_superset, allow_subset=allow_subset)
                if (newlabels := self.labels) is not None and (obs := coords.obs) is not None:
                    newlabels = tuple(self._labels[ob] for ob in obs)
                new_space = type(self)(coords, limits=self._limits_dict, binning=self.binning, label=newlabels)

        return new_space

    def with_coords(
        self,
        coords: ZfitOrderableDimensional,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> Space:
        """Create a new :py:class:`~zfit.Space` with reordered observables and/or axes.

        The behavior is that _at least one coordinate (obs or axes) has to be set in both instances
        (the space itself or in `coords`). If both match, observables is taken as the defining coordinate.
        The space is sorted according to the defining coordinate and the other coordinate is sorted as well.
        If either the space did not have the "weaker coordinate" (e.g. both have observables, but only coords
        has axes), then the resulting Space will have both.
        If both have both coordinates, obs and axes, and sorting for obs results in non-matchin axes results
        in axes being dropped.

        Args:
            coords: An instance of :py:class:`Coordinates`
            allow_superset: If `False` and a strict superset is given, an error is raised
            allow_subset: If `False` and a strict subset is given, an error is raised

        Returns:
            :py:class:`~zfit.Space`:
        Raises:
            CoordinatesUnderdefinedError: if neither both obs or axes are specified.
            CoordinatesIncompatibleError: if `coords` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if self.obs is not None and coords.obs is not None:
            new_space_obs = self.with_obs(coords.obs, allow_superset=allow_superset, allow_subset=allow_subset)

            if coords.axes is not None:  # use this axes: first drop the other one
                if new_space_obs.axes is not None:
                    new_space_obs = new_space_obs.with_axes(None)
                # filter in case there are super/subsets
                coords_axes = coords.with_obs(
                    new_space_obs.obs,
                    allow_superset=allow_superset,
                    allow_subset=allow_subset,
                )
                new_space_obs = new_space_obs.with_axes(coords_axes.axes)  # are the same or self.axes is None
            new_space = new_space_obs

        elif self.axes is not None and coords.axes is not None:
            new_space_axes = self.with_axes(coords.axes, allow_superset=allow_superset, allow_subset=allow_subset)
            if coords.obs is not None:
                # filter in case there are super/subsets
                coords_obs = coords.with_axes(
                    new_space_axes.axes,
                    allow_superset=allow_superset,
                    allow_subset=allow_subset,
                )
                new_space_axes = new_space_axes.with_obs(coords_obs.obs)
            new_space = new_space_axes
        else:
            msg = f"Neither the axes nor the obs are specified in both objects" f" {self} and {coords}"
            raise CoordinatesUnderdefinedError(msg)

        return new_space

    def with_autofill_axes(self, overwrite: bool = False) -> zfit.Space:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. `overwrite` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and `overwrite` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and `overwrite` is False.
        """
        new_coords = self.coords.with_autofill_axes(overwrite=overwrite)
        # new_space = self.with_coords(new_coords)
        return self.copy(axes=new_coords.axes) if self.axes is None or overwrite else self

    def get_subspace(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,
    ) -> Space:
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs: Observables of the subspace to return.
            axes: Axes of the subspace to return.
            name: Human readable names

        Returns:
            A space containing only a subspace (and sublimits etc.)
        """
        del name  # should be label
        if obs is not None and axes is not None:
            msg = "Cannot specify `obs` *and* `axes` to get subspace."
            raise ValueError(msg)
        if axes is None and obs is None:
            msg = "Either `obs` or `axes` has to be specified and not None"
            raise ValueError(msg)

        # try to use observables to get index
        obs = self._check_convert_input_obs(obs=obs, allow_none=True)
        axes = self._check_convert_input_axes(axes=axes, allow_none=True)
        if obs is not None:
            limits_dict = self.get_limits(obs=obs)
            new_coords = self.coords.with_obs(obs, allow_subset=True, allow_superset=True)
        else:
            limits_dict = self.get_limits(axes=axes)
            new_coords = self.coords.with_axes(axes=axes, allow_subset=True, allow_superset=True)
        if (newlabels := self.labels) is not None and (obs := new_coords.obs) is not None:
            newlabels = tuple(self._labels[ob] for ob in obs)
        return type(self)(obs=new_coords, limits=limits_dict, label=newlabels)

    @fail_not_rect
    def _legacy_area(self) -> float:
        """Return the total area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        return calculate_rect_area(rect_limits=self._rect_limits_tf)

    def with_binning(self, binning):
        return self.copy(binning=binning)

    # Operators

    def copy(self, **overwrite_kwargs) -> zfit.Space:
        """Create a new :py:class:`~zfit.Space` using the current attributes and overwriting with
        `overwrite_overwrite_kwargs`.

        Args:
            name: The new name. If not given, the new instance will be named the same as the
                current one.
            **overwrite_kwargs:

        Returns:
            :py:class:`~zfit.Space`
        """
        kwargs = {
            "name": self.name,
            "limits": self._limits_dict,
            "binning": self.binning,
            "axes": self.axes,
            "obs": self.obs,
            "label": self.labels,
        }
        kwargs.update(overwrite_kwargs)
        if set(overwrite_kwargs) - set(kwargs):
            msg = f"Not usable keys in `overwrite_kwargs`: {set(overwrite_kwargs) - set(kwargs)}"
            raise KeyError(msg)
        kwargs.get("binning")
        return self.__class__(**kwargs)

    def _inside(self, x, guarantee_limits):
        del guarantee_limits  # not used
        xs_inside = []
        obs_in_use = self.obs is not None
        limits_dict = self._limits_dict["obs" if obs_in_use else "axes"]
        for coords, limit in limits_dict.items():
            reorder_kwargs = {"func_obs" if obs_in_use else "func_axes": coords}
            x_sub = self.reorder_x(x, **reorder_kwargs)
            x_inside = limit.inside(x_sub)
            xs_inside.append(x_inside)
        return znp.all(xs_inside, axis=0)

    @property  # TODO(1.0): deprecate limits, see also https://github.com/zfit/zfit/discussions/533
    # @deprecated(date=None, instructions="use lower, upper, limits, area; for transition, see also the announcement: https://github.com/zfit/zfit/discussions/533")
    @fail_not_rect
    def limit1d(self) -> tuple[float, float]:
        """Simplified limits getter for 1 obs, 1 limit only: return the tuple(lower, upper).

        Returns:
            So :code:`lower, upper = space.limit1d` for a simple, 1 obs limit.

        Raises:
            RuntimeError: if the conditions (n_obs or n_limits) are not satisfied.
        """
        if self.n_obs > 1:
            msg = f"Cannot call `limit1d, as `Space` has more than one observables: {self.n_obs}"
            raise RuntimeError(msg)
        if self._depr_n_limits > 1:
            msg = f"Cannot call `limit1d, as `Space` has several limits: {self._depr_n_limits}"
            raise RuntimeError(msg)
        lower, upper = self.rect_limits
        return lower[0][0], upper[0][0]

    @classmethod
    @deprecated(
        date=None,
        instructions="Use directly the class to create a Space. E.g. zfit.Space(axes=(0, 1), ...)",
    )
    def from_axes(
        cls,
        axes: ztyping.AxesTypeInput,  # noqa: ARG003
        limits: ztyping.LimitsTypeInput | None = None,  # noqa: ARG003
        rect_limits=None,  # noqa: ARG003
        name: str | None = None,  # noqa: ARG003
    ) -> zfit.Space:
        """Create a space from `axes` instead of from `obs`.

        Args:
            rect_limits:
            axes:
            limits:
            name:

        Returns:
            :py:class:`~zfit.Space`
        """
        msg = "from_axes is not needed anymore, create a Space directly."
        raise BreakingAPIChangeError(msg)


def extract_limits_from_dict(limits_dict, obs=None, axes=None):
    if (obs is None) and (axes is None):
        msg = "Need to specify at least one, obs or axes."
        raise ValueError(msg)
    if (obs is not None) and (axes is not None):
        axes = None  # obs has precedency
    if obs is None:
        obs_in_use = False
        coords_to_extract = axes
    else:
        obs_in_use = True
        coords_to_extract = obs
    coords_to_extract = convert_to_container(coords_to_extract)
    coords_to_extract = set(coords_to_extract)

    limits_to_eval = {}
    limit_dict = limits_dict["obs" if obs_in_use else "axes"].items()
    keys_sorted = sorted(limit_dict, key=lambda x: len(x[0]), reverse=True)
    for key_coords, limit in keys_sorted:
        coord_intersec = frozenset(key_coords).intersection(coords_to_extract)
        if not coord_intersec:  # this limit does not contain any requested obs
            continue
        if coord_intersec == frozenset(key_coords):
            if isinstance(limit, ZfitOrderableDimensional):  # drop coordinates if given
                limit = limit.with_axes(None) if obs_in_use else limit.with_obs(None)
            limits_to_eval[key_coords] = limit
        else:
            coord_limit = [coord for coord in key_coords if coord in coord_intersec]
            kwargs = {"obs" if obs_in_use else "axes": coord_limit}
            try:
                sublimit = limit.get_subspace(**kwargs)
            except InvalidLimitSubspaceError:
                msg = f"Cannot extract {coord_intersec} from limit {limit}."
                raise InvalidLimitSubspaceError(msg) from None
            sublimit_coord = limit.obs if obs_in_use else limit.axes
            if isinstance(sublimit, ZfitOrderableDimensional):  # drop coordinates if given
                sublimit = sublimit.with_axes(None) if obs_in_use else sublimit.with_obs(None)
            limits_to_eval[sublimit_coord] = sublimit
            coords_to_extract -= coord_intersec
    return limits_to_eval


def add_spaces(*spaces: Iterable[ZfitSpace], name=None):
    """Add two spaces and merge their limits if possible or return False.

    Args:
        spaces:

    Returns:
        Union[None, :py:class:`~zfit.Space`, bool]:

    Raises:
        LimitsIncompatibleError: if limits of the `spaces` cannot be merged because they overlap
    """
    # spaces = convert_to_container(spaces)
    if not all(isinstance(space, ZfitSpace) for space in spaces):
        msg = f"Can only add type ZfitSpace, not {spaces}"
        raise TypeError(msg)
    return MultiSpace(spaces, name=name)


def get_coord(space, obs_in_use=True):
    if obs_in_use:
        return space.obs
    else:
        return space.axes


def combine_spaces(*spaces: Iterable[Space]):
    """Combine spaces with different `obs` and `limits` to one `space`.

    Checks if the limits in each obs coincide *exactly*. If this is not the case, the combination
    is not unambiguous and `False` is returned

    Args:
        spaces:

    Returns:
        Returns False if the limits don't coincide in one or more obs. Otherwise
            return the :py:class:`~zfit.Space` with all obs from `spaces` sorted by the order of `spaces` and with the
            combined limits.
    Raises:
        ValueError: if only one space is given
        LimitsIncompatibleError: If the limits of one or more spaces (or within a space) overlap
        LimitsNotSpecifiedError: If the limits for one or more obs but not all are None or False.
    """
    spaces = convert_to_container(spaces, container=tuple)
    # if len(spaces) <= 1:
    #     return spaces
    # raise ValueError("Need at least two spaces to test limit consistency.")  # TODO: allow? usecase?

    common_obs_ordered = common_obs(spaces=spaces)
    common_axes_ordered = common_axes(spaces=spaces)
    all_spaces_binned = all(space.is_binned for space in spaces)
    all_spaces_unbinned = not any(space.is_binned for space in spaces)
    if not (all_spaces_binned or all_spaces_unbinned):
        msg = (
            f"Some spaces are binned {[s for s in spaces if s.is_binned]}"
            f" while others are not {[s for s in spaces if not s.is_binned]}. Cannot mix."
        )
        raise ValueError(msg)

    all_labels_map = {}
    for space in spaces:
        if (lab := space._labels) is not None:
            all_labels_map.update(lab)
    if all_spaces_binned:
        binnings_ordererd = []
        for ob in common_obs_ordered:
            for space in spaces:
                if ob in space.obs:
                    binning = space.binning[ob]
                    if binning not in binnings_ordererd:
                        binnings_ordererd.append(binning)

        binning = Binnings(binnings_ordererd)
    else:
        binning = None
    using_obs = bool(common_obs_ordered)
    common_coords_ordered = common_obs_ordered if using_obs else common_axes_ordered

    # sort the spaces
    if using_obs:
        spaces = tuple(space.with_obs(common_obs_ordered, allow_superset=True) for space in spaces)
        all_coords = [space.obs for space in spaces]
    elif common_axes_ordered:
        spaces = tuple(space.with_axes(common_axes_ordered, allow_superset=True) for space in spaces)
        all_coords = [space.axes for space in spaces]
    else:
        msg = "Neither `obs` nor `axes` exist in all spaces."
        raise CoordinatesUnderdefinedError(msg)

    all_limits_false = all(space.limits_are_false for space in spaces)
    all_limits_not_set = all(not space.limits_are_set for space in spaces)
    has_limits = [space.has_limits for space in spaces]
    if all_limits_false:
        limits = False
    elif all_limits_not_set:
        limits = None
    elif not all(has_limits):
        msg = "Limits either have to be set, not set, or False for all spaces to be combined."
        raise LimitsNotSpecifiedError(msg)
    else:
        space_combinations = tuple(itertools.product(*spaces))
        if len(space_combinations) > 1:  # there are MultiSpaces in there
            all_combinations = []
            for spa in space_combinations:
                with suppress(LimitsIncompatibleError):
                    all_combinations.append(combine_spaces(*spa))
                if not all_combinations:
                    msg = f"The limits of {spaces} are all not compatible to be combined."
                    raise LimitsIncompatibleError(msg)
            # filter, as can be False: non-overlapping limits e.g. if we have two MultiSpace
            [space for space in all_combinations if space is not False]

            return MultiSpace(
                spaces=all_combinations,
                obs=common_obs_ordered if common_obs_ordered else None,
                axes=common_axes_ordered if common_axes_ordered else None,
            )
        # TODO: spaces that have multidim limits?
        limits_dict = {}

        non_unique_coords = set()
        unique_coords = set()
        for coord in common_coords_ordered:
            if sum(coord in coords for coords in all_coords) > 1:
                non_unique_coords.add(coord)
            else:
                unique_coords.add(coord)

        for coord in common_coords_ordered:
            if coord in unique_coords:
                space = next(space for space in spaces if coord in get_coord(space, using_obs))
                space = space.get_subspace(
                    obs=unique_coords if using_obs else None,
                    axes=None if using_obs else unique_coords,
                )
                limits_dict.update(space.get_limits()["obs" if using_obs else "axes"])
                for coord in get_coord(space, using_obs):
                    unique_coords.remove(coord)
            elif coord in non_unique_coords:
                non_unique_spaces = [space for space in spaces if coord in get_coord(space, using_obs)]
                common_coords_non_unique = list(
                    set.intersection(*(set(get_coord(space, using_obs)) for space in non_unique_spaces))
                )
                # do the below to check if we can take the subspace
                non_unique_subspaces = [
                    space.get_subspace(
                        obs=common_coords_non_unique if using_obs else None,
                        axes=None if using_obs else common_coords_non_unique,
                    )
                    for space in non_unique_spaces
                ]

                # TODO compare limits
                any_non_equal = any(non_unique_subspaces[0] != space for space in non_unique_subspaces[1:])
                if any_non_equal:
                    msg = (
                        f"Limits in coord {common_coords_non_unique} do not match for spaces" f" {non_unique_subspaces}"
                    )
                    raise LimitsIncompatibleError(msg)

                non_unique_subspace = non_unique_subspaces[0]
                limits_dict.update(non_unique_subspace.get_limits()["obs" if using_obs else "axes"])
                for coord in get_coord(non_unique_subspace, using_obs):
                    non_unique_coords.remove(coord)
            else:
                pass  # fine, since it is already satisfied by a space
        #
        # for coord in common_coords_ordered:
        #     space_with_coord = [space for space in spaces if coord in get_coord(space, using_obs)]
        #     assert not any(isinstance(space, MultiSpace) for space in space_with_coord), "bug, should be caught before."
        #     assert space_with_coord, "empty, cannot be. This is a bug."
        #     limits_coord = []
        #     for space in space_with_coord:
        #         if type(space) == Space:  # has to be the exact type, we use an implementation detail here
        #             limits_coord.append(space.get_limits(obs=coord if using_obs else None,
        #                                                  axes=coord if not using_obs else None))
        #         else:
        #             raise WorkInProgressError
        #             # limits_coord.append(space.with_obs(obs=coord) if using_obs else space.with_axes(axes=coord))
        #     any_non_equal = any([limits_coord[0][(coord,)] != limit[(coord,)] for limit in limits_coord[1:]])
        #     if any_non_equal:
        #         raise LimitsIncompatibleError(f"Limits in coord {coord} do not match for spaces {limits_coord}")
        #     limits_dict.update(limits_coord[0])

        limits = {"obs" if using_obs else "axes": limits_dict}

    # all_lower = []
    # all_upper = []
    #
    # # create the lower and upper limits with all obs replacing missing dims with None
    # # With this, all limits have the same length
    # # TODO?
    # # if limits_overlap(spaces=spaces, allow_exact_match=True):
    # #     raise LimitsIncompatibleError("Limits overlap")
    #
    # for space in flatten_spaces(spaces):
    #     if space.limits is None:
    #         continue
    #     lowers, uppers = space.limits
    #     lower = [tuple(low[space.obs.index(ob)] for low in lowers) if ob in space.obs else None for ob in all_obs]
    #     upper = [tuple(up[space.obs.index(ob)] for up in uppers) if ob in space.obs else None for ob in all_obs]
    #     all_lower.append(lower)
    #     all_upper.append(upper)
    #
    # def check_extract_limits(limits_spaces):
    #     new_limits = []
    #
    #     if not limits_spaces:
    #         return None
    #     for index, obs in enumerate(all_obs):
    #         current_limit = None
    #         for limit in limits_spaces:
    #             lim = limit[index]
    #
    #             if lim is not None:
    #                 if current_limit is None:
    #                     current_limit = lim
    #                 elif not np.allclose(current_limit, lim):
    #                     return False
    #         else:
    #             if current_limit is None:
    #                 raise LimitsNotSpecifiedError("Limits in obs {} are not specified".format(obs))
    #             new_limits.append(current_limit)
    #
    #     n_limits = int(np.prod(tuple(len(lim) for lim in new_limits)))
    #     new_limits_comb = [[] for _ in range(n_limits)]
    #     for limit in new_limits:
    #         for lim in limit:
    #             for i in range(int(n_limits / len(limit))):
    #                 new_limits_comb[i].append(lim)
    #
    #     new_limits = tuple(tuple(limit) for limit in new_limits_comb)
    #     return new_limits

    # new_lower = check_extract_limits(all_lower)
    # new_upper = check_extract_limits(all_upper)
    # assert not (new_lower is None) ^ (new_upper is None), "Bug, please report issue. either both are defined or None."
    # if new_lower is None:
    #     limits = None
    # elif new_lower is False:
    #     return False
    # else:
    #     limits = (new_lower, new_upper)

    return Space(
        obs=common_obs_ordered if using_obs else None,
        axes=None if using_obs else common_axes_ordered,
        binning=binning,
        limits=limits,
        label=tuple(all_labels_map[ob] for ob in common_coords_ordered) if using_obs else None,
    )


def less_equal_space(space1, space2, allow_graph=True):
    return compare_multispace(
        space1=space1,
        space2=space2,
        comparator=lambda limit1, limit2: limit1.less_equal(limit2, allow_graph=allow_graph),
    )


def equal_space(space1, space2, allow_graph=True):
    has_any = True
    with suppress(Exception):
        has_any = np.any(space1.ANY in np.array(space1.v1.limits)) or np.any(space2.ANY in np.array(space2.v1.limits))

    if has_any or allow_graph or isinstance(space1, MultiSpace) or isinstance(space2, MultiSpace):
        return compare_multispace(
            space1=space1,
            space2=space2,
            comparator=lambda limit1, limit2: limit1.equal(limit2, allow_graph=allow_graph),
        )
    else:
        return compare_spaces_equal_static(space1, space2)


def compare_spaces_equal_static(space1: Space, space2: Space):
    """Compare two spaces if they have the same obs, axes, and, if a comparator is given, limits.

    It is automatically checked if the limits are set resp. are False

    Args:
        space1:
        space2:
        comparator:

    Returns:
    """
    try:
        space2obs1 = space2.with_coords(space1, allow_superset=False, allow_subset=False)
    except CoordinatesIncompatibleError:
        return False

    try:
        limits1 = np.array(space1.v1.limits)
        limits2 = np.array(space2obs1.v1.limits)
        diff = np.abs(limits1 - limits2)
    except Exception:
        return False

    return np.all(diff < 1e-8)


def compare_multispace(space1: ZfitSpace, space2: ZfitSpace, comparator: Callable):
    """Compare multiple spaces if they have the same obs, axes, and, if a comparator is given, limits.

    It is automatically checked if the limits are set resp. are False

    Args:
        space1:
        space2:
        comparator:

    Returns:
    """
    axes_not_none = space1.axes is not None and space2.axes is not None
    obs_not_none = space1.obs is not None and space2.obs is not None
    if not (axes_not_none or obs_not_none):  # if both are None
        return False

    if obs_not_none:
        if set(space1.obs) != set(space2.obs):
            return False
    elif axes_not_none and set(space1.axes) != set(space2.axes):  # axes only matter if there are no obs
        return False
    if space1.binning != space2.binning:
        return False
    # check limits
    if not space1.limits_are_set:
        return bool(not space2.limits_are_set)

    elif space1.limits_are_false:
        return bool(space2.limits_are_false)

    return compare_limits_multispace(space1, space2, comparator=comparator)


def compare_limits_multispace(space1: ZfitSpace, space2: ZfitSpace, comparator: Callable) -> bool:
    if len(space1) != len(space2):
        return False
    if not (space1.has_limits and space2.has_limits):
        return False
    space2_reordered = space2.with_coords(space1)

    comparison = []
    for space11 in space1:
        compare_spaces2 = []
        for space22 in space2_reordered:
            compare_spaces2.append(
                compare_limits_coords_dict(space11.get_limits(), space22.get_limits(), comparator=comparator)
            )
        comparison.append(compare_spaces2)
    comparison = convert_to_tensor_or_numpy(comparison, dtype=tf.bool)
    space1_matches = z.unstable.reduce_any(comparison, axis=1)  # reduce over axis containing space2, has to match with
    # at least one space2.
    space2_matches = z.unstable.reduce_any(comparison, axis=0)
    all_space1_match = z.unstable.reduce_all(space1_matches, axis=0)
    all_space2_match = z.unstable.reduce_all(space2_matches, axis=0)

    return z.unstable.logical_and(all_space1_match, all_space2_match)


def compare_limits_coords_dict(
    limits1: Mapping[str, Mapping[Iterable, ZfitLimit]],
    limits2: Mapping[str, Mapping[Iterable, ZfitLimit]],
    comparator: Callable,
    require_all_coord_types: bool = False,
) -> bool:
    if limits1.keys() != limits2.keys() and require_all_coord_types:
        return False
    equal = []
    for coord_type, limit1_dict in limits1.items():
        limit2_dict = limits2.get(coord_type)
        if limit2_dict is None:
            continue
        equal.append(compare_limits_dict(limit1_dict, limit2_dict, comparator=comparator))
    return z.unstable.reduce_all(equal)


def compare_limits_dict(dict1: Mapping, dict2: Mapping, comparator: Callable) -> bool:
    comparison = []
    limits2_to_check = dict2.copy()

    for coord, limit1 in dict1.items():
        for limit2cord, limit2 in limits2_to_check.items():
            if set(limit2cord) == set(coord):
                limit2 = limits2_to_check.pop(limit2cord)
                comparison.append(comparator(limit1, limit2))
                break

        else:  # no break, nothing matched
            return False
    return z.unstable.reduce_all(comparison)


def flatten_spaces(spaces):
    return tuple(s for space in spaces for s in space)


class MultiSpace(BaseSpace):
    def __new__(
        cls,
        spaces: Iterable[ZfitSpace],
        obs: ztyping.ObsTypeInput = None,
        binning: ztyping.BinningTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,  # noqa: ARG003
    ) -> Space | MultiSpace:
        if binning is not None:
            msg = (
                "Binning not yet implemented for MultiSpace, won't ever be. Use the new truncated, multidim PDF instead"
            )
            raise RuntimeError(msg)
        spaces, obs, axes = cls._check_convert_input_spaces_obs_axes(spaces, obs, axes)
        if len(spaces) == 1:
            return spaces[0]
        space = super().__new__(cls)

        # for the __init__ below, see there
        space._tmp_store_spaces_obs_axes = spaces, obs, axes

        return space

    @deprecated(
        date=None,
        instructions="MultiSpace is deprecated, see announcement here: https://github.com/zfit/zfit/discussions/533"
        "If you're use-case is not covered by this or you have questions, please open an issue "
        " at https://github.com/zfit/zfit/issues/new?assignees=mayou36&labels=discussion&projects=&template=behaviorunderdiscussion.md&title=%5BMultiSpace%20deprecation%5D",
    )
    def __init__(self, spaces: Iterable[ZfitSpace], obs=None, axes=None, name: str | None = None) -> None:
        # Since __new__ returns an instance of MultiSpace, __init__ is invoked. We don't want to reprocess
        # the input arguments here, so we store them above in the dummy attribute.
        del spaces, obs, axes  # not needed, we take the already preprocessed.
        spaces, obs, axes = self._tmp_store_spaces_obs_axes
        del self._tmp_store_spaces_obs_axes
        if name is None:
            name = "MultiSpace"
        super().__init__(obs, axes, name)

        self._labels = {ob: ob for ob in self.obs} if self.obs is not None else None

        self.spaces = spaces
        self.v0 = V0Space(self)

    @property
    def label(self):
        return None

    @property
    def labels(self):
        return self.obs if self.obs is not None else None

    @staticmethod
    def _initialize_space(space, spaces, obs, axes):
        space._obs = obs
        space._axes = axes
        space.spaces = spaces
        return space

    @staticmethod
    def _check_convert_input_spaces_obs_axes(spaces, obs, axes):  # TODO: do something with axes
        spaces = flatten_spaces(spaces)
        all_have_obs = all(space.obs is not None for space in spaces)
        all_have_axes = all(space.axes is not None for space in spaces)
        all_binnings_compatible = len({space.binning for space in spaces}) == 1
        n_events = [space.n_events in (spaces[0].n_events, None) for space in spaces]
        all_nevents_compatible = all(n_events)
        if not all_binnings_compatible:
            msg = "Binnings not compatible, maybe this needs to be better care taken."
            raise ValueError(msg)
        if not all_nevents_compatible:
            msg = "The number of events of the spaces do not coincide"
            raise NumberOfEventsIncompatibleError(msg)
        if all_have_axes:
            axes = spaces[0].axes if axes is None else convert_to_axes(axes)

        if all_have_obs:
            obs = spaces[0].obs if obs is None else convert_to_obs_str(obs)
            spaces = [space.with_obs(obs, allow_subset=False, allow_superset=False) for space in spaces]
            if not (all_have_axes and all(space.axes == axes for space in spaces)):  # obs coincide, axes don't -> drop
                spaces = [space.with_axes(None) for space in spaces]

        elif all_have_axes:
            if all(space.obs is None for space in spaces):
                spaces = [space.with_axes(axes, allow_superset=False, allow_subset=False) for space in spaces]
            if not (all_have_obs and all(space.obs == obs for space in spaces)):  # axes coincide, obs don't -> drop
                spaces = [space.with_obs(None) for space in spaces]

        else:
            msg = "Spaces do not have consistent obs and/or axes."
            raise SpaceIncompatibleError(msg)

        if all(space.has_limits for space in spaces):
            # check overlap, reduce common limits
            pass
        elif not any(space.has_limits for space in spaces):
            spaces = [spaces[0]]  # if all are None, then nothing to add
        else:  # some have limits, some don't -> does not really make sense (or just drop the ones without limits?)
            msg = (
                "Some spaces have limits, other don't. This behavior may change in the future "
                "to allow spaces with None to be simply ignored.\n"
                "If you prefer this behavior, please open an issue on github."
            )
            raise LimitsIncompatibleError(msg)

        spaces = tuple(spaces)

        return spaces, obs, axes

    @property
    def binning(self):
        return self.spaces[0].binning

    # noinspection PyPropertyDefinition
    @property
    @fail_not_rect
    def limits(self) -> None:
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    def rect_limits(self):
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    @fail_not_rect
    def lower(self) -> None:
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    @fail_not_rect
    def upper(self) -> None:
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    def rect_limits_np(self):
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    def rect_lower(self):
        self._raise_limits_not_implemented()

    # noinspection PyPropertyDefinition
    @property
    def rect_upper(self):
        self._raise_limits_not_implemented()

    def rect_area(self) -> float | np.ndarray | tf.Tensor:
        """Calculate the total rectangular area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        return z.reduce_sum([space._legacy_area() for space in self], axis=0)

    @property
    def rect_limits_are_tensors(self) -> bool:
        """Return True if the rectangular limits are tensors.

        If a limit with tensors is evaluated inside a graph context, comparison operations will fail.

        Returns:
            If the rectangular limits are tensors.
        """
        return all(space.limits_are_tensors for space in self)

    @property
    def has_rect_limits(self) -> bool:
        """If there are limits and whether they are rectangular."""
        return all(space.has_rect_limits for space in self.spaces)

    # noinspection PyPropertyDefinition

    @property
    def limits_are_false(self) -> bool:
        """If the limits have been set to False, so the object on purpose does not contain limits.

        Returns:
            True if limits is False
        """
        return all(space.limits_are_false for space in self.spaces)

    # noinspection PyPropertyDefinition

    @property
    def has_limits(self) -> bool:
        """Whether there are limits set and they are not false.

        Returns:
        """
        try:
            return self.limits_are_set and not self.limits_are_false
        except MultipleLimitsNotImplemented:
            return True

    @property
    def limits_are_set(self) -> bool:
        """If the limits have been set to a limit or are False.

        Returns:
            Whether the limits have been set or not.
        """
        return all(space.limits_are_set for space in self.spaces)

    @property
    def n_events(self) -> int | None:
        """Shape of the first dimension, usually reflects the number of events.

        Returns:
            Return the number of events, the dimension of the first shape. If this is > 1 or None,
                it's vectorized.
        """
        # get the first numeric n_events. Is None if a Tensor and not specified yet.
        n_events_first = [space.n_events for space in self if space.n_events is not None]
        return None if not n_events_first else n_events_first[0]

    def with_limits(
        self,
        limits: ztyping.LimitsTypeInput = None,
        rect_limits: ztyping.RectLimitsInputType | None = None,
        name: str | None = None,
    ) -> ZfitSpace:
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits: Limits to use. Can be rectangular, a function (requires to also specify `rect_limits`
                or an instance of ZfitLimit.
            rect_limits: Rectangular limits that will be assigned with the instance
            name: Human readable name

        Returns:
            Copy of the current object with the new limits.
        """
        return self.copy(
            spaces=[space.with_limits(limits=limits, rect_limits=rect_limits) for space in self],
            name=name,
        )

    @fail_not_rect
    def _legacy_area(self) -> float:
        return self.rect_area()

    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new Space that has `obs`; sorted by or set or dropped.

        The behavior is as follows:

         * obs are already set:
           * input obs are None: the observables will be dropped. If no axes are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input obs are not None: the instance will be sorted by the incoming obs. If axes or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned.
             This can be used to take a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the obs will be sorted accordingly as
             if the obs not contained in the instances obs were not in the input obs.
         * obs are not set:
           * if the input obs are None, the same object is returned.
           * if the input obs are not None, they will be set as-is and now correspond to the already
             existing axes in the object.

        Args:
            obs: Observables to sort/associate this instance with
            allow_superset: if False and a strict superset of the own observables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own observables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/observables

        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            ObsIncompatibleError: if `obs` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        spaces = [
            space.with_obs(obs, allow_superset=allow_superset, allow_subset=allow_subset) for space in self.spaces
        ]
        coords = self.coords.with_obs(obs, allow_subset=allow_subset, allow_superset=allow_superset)
        return self.copy(spaces=spaces, obs=coords.obs, axes=coords.axes)

    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new instance that has `axes`; sorted by or set or dropped.

        The behavior is as follows:

         * axes are already set:
           * input axes are None: the axes will be dropped. If no observables are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input axes are not None: the instance will be sorted by the incoming axes. If obs or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned. This can
             be used to retrieve a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the axes will be sorted accordingly as
             if the axes not contained in the instances axes were not present in the input axes.
         * axes are not set:
           * if the input axes are None, the same object is returned.
           * if the input axes are not None, they will be set as-is and now correspond to the already
             existing obs in the object.

        Args:
            axes: Axes to sort/associate this instance with
            allow_superset: if False and a strict superset of the own axeservables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own axeservables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/axes
        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            AxesIncompatibleError: if `axes` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        spaces = [
            space.with_axes(axes, allow_superset=allow_superset, allow_subset=allow_subset) for space in self.spaces
        ]
        coords = self.coords.with_axes(axes, allow_subset=allow_subset, allow_superset=allow_superset)
        return self.copy(spaces=spaces, obs=coords.obs, axes=coords.axes)

    def with_coords(
        self,
        coords: ZfitOrderableDimensional,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new :py:class:`~zfit.Space` with reordered observables and/or axes.

        The behavior is that _at least one coordinate (obs or axes) has to be set in both instances
        (the space itself or in `coords`). If both match, observables is taken as the defining coordinate.
        The space is sorted according to the defining coordinate and the other coordinate is sorted as well.
        If either the space did not have the "weaker coordinate" (e.g. both have observables, but only coords
        has axes), then the resulting Space will have both.
        If both have both coordinates, obs and axes, and sorting for obs results in non-matchin axes results
        in axes being dropped.

        Args:
            coords: An instance of :py:class:`Coordinates`
            allow_superset: If false and a strict superset is given, an error is raised
            allow_subset: If false and a strict subset is given, an error is raised

        Returns:
            :py:class:`~zfit.Space`:
        Raises:
            CoordinatesUnderdefinedError: if neither both obs or axes are specified.
            CoordinatesIncompatibleError: if `coords` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        new_spaces = [
            space.with_coords(coords, allow_superset=allow_superset, allow_subset=allow_subset) for space in self
        ]
        return type(self)(spaces=new_spaces)

    def with_autofill_axes(self, overwrite: bool = False) -> MultiSpace:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. `overwrite` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and `overwrite` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and `overwrite` is False.
        """
        spaces = [space.with_autofill_axes(overwrite=overwrite) for space in self]
        return self.copy(spaces=spaces)

    def get_subspace(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,
    ) -> MultiSpace:
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs: Observables of the subspace to return.
            axes: Axes of the subspace to return.
            name: Human readable names

        Returns:
            A space containing only a subspace (and sublimits etc.)
        """
        spaces = [space.get_subspace(obs=obs, axes=axes) for space in self.spaces]
        return self.copy(spaces=spaces, name=name)

    def copy(self, *, deep: bool = False, name: str | None = None, **overwrite_params) -> MultiSpace:
        assert not deep, "deep not explicitly implemented, should not be needed for immutable objects"
        kwargs = {
            "spaces": tuple(self),
            "obs": self.obs,
            "axes": self.axes,
        }
        kwargs.update(overwrite_params)
        kwargs["name"] = self.name if name is None else name
        return type(self)(**kwargs)

    def _raise_limits_not_implemented(self):
        msg = (
            "Limits/lower/upper not implemented for MultiSpace. This error is either caught"
            " automatically as part of the codes logic or the MultiLimit case should"
            " be considered. To do that, simply iterate through the MultiSpace, which returns"
            " a simple space. Iterating through a Spaces also works"
            "for simple spaces."
        )
        raise MultipleLimitsNotImplemented(msg)

    def _inside(self, x, guarantee_limits):
        inside_limits = [space.inside(x, guarantee_limits=guarantee_limits) for space in self]
        return znp.any(inside_limits, axis=0)  # has to be inside one limit

    def __iter__(self) -> ZfitSpace:
        yield from self.spaces

    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.has_rect_limits:
            if self.n_obs < 3 and not self.n_events > 1 and self._depr_n_limits <= 3:
                limits = [lim.v0.limits for lim in self]
            else:
                limits = "rectangular"
        else:
            limits = "functional"
        return f"<zfit {class_name} obs={self.obs}, axes={self.axes}, limits={limits}>"

    def __eq__(self, other):
        # in principle, two disjoint regions could have coinciding lower and upper limits equaling to an actually larger
        # space. However, we ignore this case and say that l1 to u1 and l2 to u2 is never the same as l1 to u2,
        # even if u1 == l2 (and they mathematically coincide).
        if not isinstance(other, MultiSpace):
            warnings.warn("Multispace limits compare never equal to Space.", stacklevel=2)
        return equal_space(self, other, allow_graph=False)

    def __le__(self, other):
        # in principle, two disjoint regions could have coinciding lower and upper limits equaling to an actually larger
        # space. However, we ignore this case and say that l1 to u1 and l2 to u2 is never the same as l1 to u2,
        # even if u1 == l2 (and they mathematically coincide).
        if not isinstance(other, MultiSpace):
            warnings.warn("Multispace limits compare never equal to Space.", stacklevel=2)
        return less_equal_space(self, other, allow_graph=False)

    def __hash__(self):
        return hash(self.spaces)


def convert_to_space(
    obs: ztyping.ObsTypeInput | None = None,
    axes: ztyping.AxesTypeInput | None = None,
    limits: ztyping.LimitsTypeInput | None = None,
    *,
    overwrite_limits: bool = False,
    one_dim_limits_only: bool = True,
    simple_limits_only: bool = True,
) -> None | ZfitSpace | bool:
    """Convert *limits* to a :py:class:`~zfit.Space` object if not already None or False.

    Args:
        obs:
        limits:
        axes:
        overwrite_limits: If `obs` or `axes` is a :py:class:`~zfit.Space` _and_ `limits` are given, return an instance
            of :py:class:`~zfit.Space` with the new limits. If the flag is `False`, the `limits` argument will be
            ignored if
        one_dim_limits_only:
        simple_limits_only:

    Returns:
        Union[:py:class:`~zfit.Space`, False, None]:

    Raises:
        OverdefinedError: if `obs` or `axes` is a :py:class:`~zfit.Space` and `axes` respectively `obs` is not `None`.
    """
    space = None

    # Test if already `Space` and handle
    if isinstance(obs, ZfitSpace):
        if axes is not None:
            msg = "if `obs` is a `Space`, `axes` cannot be defined."
            raise OverdefinedError(msg)
        space = obs
    elif isinstance(axes, ZfitSpace):
        if obs is not None:
            msg = "if `axes` is a `Space`, `obs` cannot be defined."
            raise OverdefinedError(msg)
        space = axes
    elif isinstance(limits, ZfitSpace):
        return limits
    if space is not None:
        # set the limits if given
        if limits is not None and (overwrite_limits or not space.limits_are_set):
            if isinstance(limits, ZfitSpace):  # figure out if compatible if limits is `Space`
                if not (
                    limits.obs == space.obs or (limits.axes == space.axes and limits.obs is None and space.obs is None)
                ):
                    msg = "`obs`/`axes` is a `Space` as well as the `limits`, but the " "obs/axes of them do not match"
                    raise IntentionAmbiguousError(msg)
                limits = False if limits.limits_are_false else limits.limits

            space = space.with_limits(limits=limits)
        return space

    # space is None again
    if not (obs is None and axes is None):
        # check if limits are allowed
        space = Space(obs=obs, axes=axes, limits=limits)  # create and test if valid
        if one_dim_limits_only and space.n_obs > 1 and space.has_limits:
            msg = "Limits more sophisticated than 1-dim cannot be auto-created from tuples. Use `Space` instead."
            raise LimitsUnderdefinedError(msg)
        if simple_limits_only and space.has_limits and space._depr_n_limits > 1:
            msg = "Limits with multiple limits cannot be auto-created" " from tuples. Use `Space` instead."
            raise LimitsUnderdefinedError(msg)
    return space


def check_norm(supports=None):
    if supports is None:
        supports = False
    supports = convert_to_container(supports, convert_none=True)

    def no_norm_range(func):
        """Decorator: Catch the 'norm' kwargs.

        If not None, raise `NormNotImplemented`.
        """
        parameters = inspect.signature(func).parameters
        keys = list(parameters.keys())
        norm_range_index = None
        norm_index = None
        if "norm_range" in keys:
            norm_range_index = keys.index("norm_range")
        if "norm" in keys:
            norm_index = keys.index("norm")

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            self = args[0] if len(args) > 0 else None
            norm_range = kwargs.get("norm_range")
            norm = kwargs.get("norm")

            norm_is_arg = False
            if norm_range is not None:
                norm = norm_range
            elif norm is not None:
                pass
            else:
                if norm_range_index is not None:
                    norm_is_arg = len(args) > norm_range_index
                    if norm_is_arg:
                        norm = args[norm_range_index]
                if norm_index is not None:
                    norm_is_arg = len(args) > norm_index
                    if norm_is_arg:
                        norm = args[norm_index]
            args = list(args)
            # if not norm_is_arg:  # TODO: remove? why is this here?
            #     if 'norm_range' in kwargs:
            #         kwargs['norm_range'] = False
            #     if 'norm' in kwargs:
            #         kwargs['norm'] = False

            # assume it's not supported. Switch if we find that it is supported.
            norm_not_supported = supports[0] is not True
            if isinstance(norm, ZfitSpace):
                if "space" in supports and isinstance(self, ZfitPDF) and self.space == norm:
                    norm_not_supported = False
                if "norm" in supports and isinstance(self, ZfitPDF) and self.norm == norm:
                    norm_not_supported = False
                if norm_not_supported:
                    norm_not_supported = not norm.limits_are_false
                    if norm.limits_are_false:
                        if not norm_is_arg:  # TODO: remove? why is this here?
                            if "norm_range" in kwargs:
                                kwargs["norm_range"] = False
                            if "norm" in kwargs:
                                kwargs["norm"] = False
                        elif norm_range_index is not None:
                            args[norm_range_index] = False
                        elif norm_index is not None:
                            args[norm_index] = False
            elif norm_not_supported:
                norm_not_supported = not (norm is None or norm is False)
            if norm_not_supported:
                raise NormNotImplemented()
            try:
                return func(*args, **kwargs)
            except TypeError as error:
                if "got an unexpected keyword argument 'norm_range'" in str(error):
                    kwargs.pop("norm_range")
                elif "got an unexpected keyword argument 'norm'" in str(error):
                    kwargs.pop("norm")
                else:
                    raise
                return func(*args, **kwargs)

        return new_func

    return no_norm_range


def param_args_supported(func):
    """Decorator: Catch the 'params' kwargs if not supported."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        self = args[0] if len(args) > 0 else None

        if "params" not in kwargs and self is not None:
            params = {key: znp.asarray(val) for key, val in self.params.items()}
            kwargs["params"] = params

        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "got an unexpected keyword argument 'params'" in str(error):
                kwargs.pop("params")
            else:
                raise
            return func(*args, **kwargs)

    return new_func


def no_multiple_limits(func):
    """Decorator: Catch the 'limits' kwargs.

    If it contains multiple limits, raise MultipleLimitsNotImplementedError.
    """
    parameters = inspect.signature(func).parameters
    if "limits" in (keys := list(parameters.keys())):
        limits_index = keys.index("limits")
    else:
        return func  # no limits as parameters -> no problem

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        limits_is_arg = len(args) > limits_index
        limits = args[limits_index] if limits_is_arg else kwargs["limits"]

        if limits._depr_n_limits > 1:
            raise MultipleLimitsNotImplemented
        return func(*args, **kwargs)

    return new_func


@deprecated_norm_range
def supports(
    *,
    norm: bool | str | Iterable[str] | None = None,
    multiple_limits: bool | None = None,
) -> Callable:
    """Decorator: Add (mandatory for some methods) on a method to control what it can handle.

    If any of the flags is set to False, it will check the arguments and, in case they match a flag
    (say if a *norm* is passed while the *norm* flag is set to `False`), it will
    raise a corresponding exception (in this example a `NormRangeNotImplementedError`) that will
    be catched by an earlier function that knows how to handle things.

    Args:
        norm: If False, no norm_range argument will be passed through resp. will be `None`.
            Other options include `'space'` or `'norm'`, which will check if the norm is equal to
            the space or norm of the PDF. If they are, it is assumed to be supported.
        multiple_limits: If False, only simple limits are to be expected and no iteration is
            therefore required.
    """
    if norm is None:
        norm = False
    if multiple_limits is None:
        multiple_limits = False

    decorator_stack = []
    if not multiple_limits:
        decorator_stack.append(no_multiple_limits)

    if norm is not None:  # check True. Could also be a str
        decorator_stack.append(check_norm(norm))

    decorator_stack.append(param_args_supported)

    def create_deco_stack(func):
        for decorator in reversed(decorator_stack):
            func = decorator(func)
        func.__wrapped__ = supports
        return func

    return create_deco_stack


def contains_tensor(objects):
    tensor_found = tf.is_tensor(objects)
    with suppress(TypeError):
        for obj in objects:
            if tensor_found:
                break
            tensor_found += contains_tensor(obj)
    return tensor_found


def shape_np_tf(objects):
    return tuple(tf.convert_to_tensor(objects).shape.as_list()) if contains_tensor(objects) else np.shape(objects)


def limits_consistent(spaces: Iterable[zfit.Space]):
    """Check if space limits are the *exact* same in each obs they are defined and therefore are compatible.

    In this case, if a space has several limits, e.g. from -1 to 1 and from 2 to 3 (all in the same observable),
    to be consistent with this limits, other limits have to have (in this obs) also the limits
    from -1 to 1 and from 2 to 3. Only having the limit -1 to 1 _or_ 2 to 3 is considered _not_ consistent.

    This function is useful to check if several spaces with *different* observables can be _combined_.

    Args:
        spaces:

    Returns:
    """
    try:
        _ = combine_spaces(*spaces)
    except LimitsIncompatibleError:
        return False
    else:
        return True


def add_spaces_old(spaces: Iterable[zfit.Space]):
    """Add two spaces and merge their limits if possible or return False.

    Args:
        spaces:

    Returns:
        Union[None, :py:class:`~zfit.Space`, bool]:

    Raises:
        LimitsIncompatibleError: if limits of the `spaces` cannot be merged because they overlap
    """
    spaces = convert_to_container(spaces)
    if not all(isinstance(space, ZfitSpace) for space in spaces):
        msg = "Cannot only add type ZfitSpace"
        raise TypeError(msg)
    if len(spaces) <= 1:
        msg = "Need at least two spaces to be added."
        raise ValueError(msg)  # TODO: allow? usecase?
    obs = frozenset(frozenset(space.obs) for space in spaces)

    if len(obs) != 1:
        return False

    obs1 = spaces[0].obs
    spaces = [space.with_obs(obs=obs1) if space.obs != obs1 else space for space in spaces]

    if limits_overlap(spaces=spaces, allow_exact_match=True):
        msg = "Limits of spaces overlap, cannot merge spaces."
        raise LimitsIncompatibleError(msg)

    lowers = []
    uppers = []
    for space in spaces:
        if not space.limits_are_set:
            continue
        for lower, upper in space:
            for other_lower, other_upper in zip(lowers, uppers):
                lower_same = np.allclose(lower, other_lower)
                upper_same = np.allclose(upper, other_upper)
                assert not lower_same ^ upper_same, "Bug, please report as issue. limits_overlap did not catch right."
                if lower_same and upper_same:
                    break
            else:
                lowers.append(lower)
                uppers.append(upper)
    lowers = tuple(lowers)
    uppers = tuple(uppers)
    limits = None if len(lowers) == 0 else (lowers, uppers)
    return zfit.Space(obs=spaces[0].obs, limits=limits)

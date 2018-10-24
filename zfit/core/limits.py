from __future__ import print_function, division, absolute_import

import inspect
from typing import Tuple, Union

import numpy as np

from zfit.util.exception import NormRangeNotImplementedError


class Range(object):
    FULL = object()  # unique reference

    def __init__(self, limits, dims=None):
        """Range holds limits and specifies dimension.

        Args:
            limits (tuple): A 1 dimensional tuple is interpreted as a list of 1 dimensional limits
                (lower1, upper1, lower2, upper2,...). Simple example: (-4, 3) means limits from
                -4 to 3.
                Higher dimensions are created with tuples of the shape (n_dims, n_(lower, upper))
                where the number of lower, upper pairs can vary in each dimension.

                Example: ((-1, 5), (-4, 1, 2, 5)) translates to: first dimension goes from -1 to 5,
                    the second dimension from -4 to 1 and from 2 to 5.

        Returns:
            Range: Returns the range object itself
        """
        self._area = None
        self._set_limits_and_dims(limits, dims)

    def _set_limits_and_dims(self, limits, dims):
        # TODO all the conversions come here
        limits, inferred_dims, has_none = self.sanitize_limits(limits)
        assert len(limits) == len(inferred_dims)
        dims = self.sanitize_dims(dims)
        if dims is Range.FULL:
            dims = inferred_dims
        if dims is None:
            if has_none:
                dims = inferred_dims
            else:
                raise ValueError(
                    "Due to safety: no dims provided, no Nones in limits. Provide dims.")
        else:  # only check if dims from user input
            if len(dims) != len(limits):
                raise ValueError("Dims {dims} and limits {lims} "
                                 "have different number of axis.".format(dims=dims, lims=limits))

        self._limits = limits
        self._dims = dims

    @property
    def n_dims(self):
        return len(self.dims)

    @staticmethod
    def sanitize_limits(limits):
        inferred_dims = []
        sanitized_limits = []
        has_none = False
        for i, dim in enumerate(limits):
            if dim is not None:
                sanitized_limits.append(dim)
                inferred_dims.append(i)
            else:
                has_none = True
        if len(np.shape(sanitized_limits)) == 1:
            are_scalars = [np.shape(l) == () for l in sanitized_limits]
            all_scalars = all(are_scalars)
            all_tuples = not any(are_scalars)
            if not (all_scalars or all_tuples):
                raise ValueError("Invalid format for limits: {}".format(limits))
            elif all_scalars:
                sanitized_limits = (tuple(sanitized_limits),)
                inferred_dims = (0,)
        sanitized_limits = Range.sort_limits(sanitized_limits)
        inferred_dims = Range.sanitize_dims(inferred_dims)
        return sanitized_limits, inferred_dims, has_none

    @staticmethod
    def sanitize_dims(dims):
        if dims is None or dims is Range.FULL:
            return dims

        if len(np.shape(dims)) == 0:
            dims = (dims,)
        sorted_dims = tuple(sorted(dims))
        # print("DEBUG", sorted dims", sorted_dims)
        if len(np.shape(dims)) == 0:
            sorted_dims = (sorted_dims,)
        return sorted_dims

    @property
    def area(self):
        """Return the total area of all the limits and dims. Useful, for example, for MC integration."""
        if self._area is None:
            self._calculate_save_area()
        return self._area

    def _calculate_save_area(self):
        area = 1.
        for dims in self:
            sub_area = 0
            for lower, upper in iter_limits(dims):
                sub_area += upper - lower
            area *= sub_area
        self._area = area
        return area

    @property
    def dims(self):
        return self._dims

    def as_tuple(self):
        return self._limits

    def as_array(self):
        return np.array(self._limits)

    def subspace(self, dims: Tuple[int]) -> 'Range':
        """Return an instance of Range containing only a subspace (`dims`) of the instance"""
        sub_range = Range(limits=self.idims(dims), dims=dims)
        return sub_range

    def get_boundaries(self):
        """Return a lower and upper boundary tuple containing all possible combinations.

        The limits given in the tuple form are converted to two tuples: one containing all of the
        possible combinations of the lower limits and the other one containing all possible
        combinations of the upper limits. This is useful to evaluate integrals.
        Example: the tuple ((low1_a, up1_b), (low2_a, up2_b, low2_c, up2_d, low2_e, up2_f))
            transforms to two tuples:
            lower: ((low1_a, low2_a), (low1_a, low2_c), (low1_a, low2_e))
            upper: ((up1_b, up2_b), (up1_b, up2_d), (up1_b, up2_f))

        Returns:
            tuple(lower, upper): as defined in the example
        """
        # print("DEBUG":, tuple", self.as_tuple())
        lower, upper = Range.combine_boundaries(self.as_tuple())
        return tuple(lower), tuple(upper)

    @staticmethod
    def combine_boundaries(limits):
        if len(limits) == 0:
            return [], []
        lower_limits = []
        upper_limits = []
        for lower, upper in iter_limits(limits[0]):
            other_lower = None  # check if for loop gets executed
            for other_lower, other_upper in zip(*Range.combine_boundaries(limits[1:])):
                lower_limits.append(tuple([lower] + list(other_lower)))
                upper_limits.append(tuple([upper] + list(other_upper)))
            if other_lower is None:  # we're in the last axis
                lower_limits.append((lower,))
                upper_limits.append((upper,))
        return tuple(lower_limits), tuple(upper_limits)

    @classmethod
    def from_boundaries(cls, lower, upper, dims=None):
        """Create a Range instance from a lower, upper limits pair. Opposite of Range.get_boundaries()

        Args:
            lower (tuple):
            upper (tuple):
            dims (tuple(int)): The dimensions the limits belong to.
        Returns:
            zfit.core.limits.Range:
        """
        # TODO: make use of Nones?
        limits = cls.extract_boundaries(lower, upper)
        return Range(limits=limits, dims=dims)

    @staticmethod
    def extract_boundaries(lower, upper):
        lower = Range._add_dim_if_scalars(lower)
        upper = Range._add_dim_if_scalars(upper)

        if not np.shape(lower) == np.shape(upper) or len(np.shape(lower)) != 2:
            raise ValueError("lower {} and upper {} have to have the same (n_limits, n_dims) shape."
                             "Currently lower shape: {} upper shape: {}"
                             "".format(lower, upper, np.shape(lower), np.shape(upper)))
        limits = [[] for _ in range(len(lower[0]))]
        for lower_vals, upper_vals in zip(lower, upper):
            for i, (lower_val, upper_val) in enumerate(zip(lower_vals, upper_vals)):
                limits[i].extend((lower_val, upper_val))

        # TODO: do some checks here?

        # make boundaries unique
        return Range.sort_limits(limits)

    @staticmethod
    def _add_dim_if_scalars(values):
        if len(np.shape(values)) == 0:
            values = (values,)
        if len(np.shape(values)) == 1:
            if all(np.shape(v) == () for v in values):
                values = (values,)
        return values

    @staticmethod
    def sort_limits(limits):
        limits = tuple(tuple(sorted(set(vals))) for vals in limits)
        return Range._add_dim_if_scalars(limits)

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Comparison between other types than Range objects currently not "
                            "supported")
        if self.dims != other.dims:
            return False
        for dim, other_dim in zip(self, other):
            for lower, upper in iter_limits(dim):
                is_smaller = False
                for other_lower, other_upper in iter_limits(other_dim):
                    is_smaller = other_lower <= lower and upper <= other_upper
                    if is_smaller:
                        break
                if not is_smaller:
                    return False
        return True

    def __gt__(self, other):
        return other < self

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Comparison between other types than Range objects currently not "
                            "supported")
        if self.dims != other.dims:
            return False
        return self.as_tuple() == other.as_tuple()

    def __getitem__(self, key):
        try:
            limits = tuple(self.as_tuple()[axis] for axis in key)
        except TypeError:
            limits = self.as_tuple()[key]
        return limits

    def idims(self, dims):
        if not hasattr(dims, "__len__"):
            dims = (dims,)
        limits_by_dims = tuple([self[self.dims.index(dim)] for dim in dims])
        return limits_by_dims


def convert_to_range(limits, dims=None) -> Union[Range, bool, None]:
    """Convert *limits* to a Range object if not already None or False.

    Args:
        limits (Union[Tuple[float, float], zfit.core.limits.Range]):
        dims (Union[Range, False, None]):

    Returns:
        Union[Range, False, None]:
    """
    if limits is None:
        return limits
    elif limits is False:
        return limits
    elif isinstance(limits, Range):
        return limits
    else:
        return Range(limits, dims=dims)


def iter_limits(limits):
    """Returns (lower, upper) for an iterable containing several such pairs

    Args:
        limits (iterable): A 1-dimensional iterable containing an even number of values. The odd
            values are takes as the lower limit while the even values are taken as the upper limit.
            Example: [a_lower, a_upper, b_lower, b_upper]

    Returns:
        iterable(tuples(lower, upper)): Returns an iterable containing the lower, upper tuples.
            Example (from above): [(a_lower, a_upper), (b_lower, b_upper)]

    Raises:
        ValueError: if limits does not contain an even number of elements.
    """
    if not len(limits) % 2 == 0:
        raise ValueError("limits has to be from even length, not: {}".format(limits))
    return zip(limits[::2], limits[1::2])


def no_norm_range(func):
    """Decorator: Catch the 'norm_range' kwargs. If not None, raise NormRangeNotImplementedError."""
    parameters = inspect.signature(func).parameters
    keys = list(parameters.keys())
    if 'norm_range' in keys:
        norm_range_index = keys.index('norm_range')
    else:
        norm_range_index = None

    def new_func(*args, **kwargs):
        norm_range_not_false = not (kwargs.get('norm_range') is None or kwargs.get('norm_range') is False)
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

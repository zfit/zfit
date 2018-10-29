from __future__ import print_function, division, absolute_import

import inspect
from typing import Tuple, Union

import numpy as np

from zfit.util.exception import NormRangeNotImplementedError, MultipleLimitsNotImplementedError


class Range(object):
    FULL = object()  # unique reference
    ANY = object()
    ANY_LOWER = object()  # TODO: need different upper, lower?
    ANY_UPPER = object()
    __HASH_DELIMINATOR = object()

    def __init__(self, *, limits=None, lower=None, upper=None, dims=None,
                 convert_none=False):  # TODO: change lower, upper?
        """Range holds limits and specifies dimension.

        Args:

        Returns:
            Range: Returns the range object itself
        """
        # input validation
        limits_valid_specified = limits is not None and (lower is None and upper is None)
        boundaries_valid_specified = limits is None and (lower is not None and upper is not None)
        if not (limits_valid_specified or boundaries_valid_specified):
            raise ValueError("Invalid argument signature! Either specify the limits OR the lower and upper boundaries.")
        if dims is None:
            raise ValueError("`dims` is None. Has to be specified.")

        self._area = None
        self._area_by_boundaries = None
        self._boundaries = None  # gets set below
        self._dims = None  # gets set below
        if dims is None:
            raise ValueError("DIMS IS NONE")
        if limits is not None:
            limits, _ = self.sanitize_limits(limits, dims=dims, repl_none=convert_none)
            lower, upper = self.boundaries_from_limits(limits)
        self._set_boundaries_and_dims(lower=lower, upper=upper, dims=dims, repl_none=convert_none)

    @classmethod
    def from_boundaries(cls, lower, upper, dims=None, *, convert_none=False):
        """Create a Range instance from a lower, upper limits pair. Opposite of Range.get_boundaries()

        Args:
            lower (tuple):
            upper (tuple):
            dims (tuple(int)): The dimensions the limits belong to.
        Returns:
            zfit.core.limits.Range:
        """
        # TODO: make use of Nones?
        return Range(lower=lower, upper=upper, dims=dims, convert_none=convert_none)

    @classmethod
    def from_limits(cls, limits, dims, *, convert_none=False):
        """            limits (tuple): A 1 dimensional tuple is interpreted as a list of 1 dimensional limits
                (lower1, upper1, lower2, upper2,...). Simple example: (-4, 3) means limits from
                -4 to 3.
                Higher dimensions are created with tuples of the shape (n_dims, n_(lower, upper))
                where the number of lower, upper pairs can vary in each dimension.

                Example: ((-1, 5), (-4, 1, 2, 5)) translates to: first dimension goes from -1 to 5,
                    the second dimension from -4 to 1 and from 2 to 5.
        """
        return Range(limits=limits, dims=dims)

    def __len__(self):
        return len(self.get_boundaries()[0])

    @staticmethod
    def sanitize_boundaries(lower, upper, dims=None, convert_none=False):
        """Sanitize (add dim, replace None, check length...)

        Args:
            lower (iterable):
            upper (iterable):
            dims (iterable):
            convert_none (bool):

        Returns:
            lower, upper, inferred_dims: each one is a 2-d tuple containing the limits and a tuple
                with the inferred axis
        """
        inferred_dims = None
        # input check
        if np.shape(lower) == ():
            lower = (lower,)
        if np.shape(upper) == ():
            upper = (upper,)

        if not len(lower) == len(upper):
            raise ValueError("lower and upper bounds do not have the same length:"
                             "\nlower: {}"
                             "\nupper: {}".format(lower, upper))
        dims = Range.sanitize_dims(dims, allow_none=True)

        new_lower = []
        new_upper = []
        for bounds, new_bounds, none_repl in zip((lower, upper), (new_lower, new_upper),
                                                 (Range.ANY_LOWER, Range.ANY_UPPER)):
            are_scalars = [np.shape(l) == () for l in bounds]
            all_scalars = all(are_scalars)
            all_tuples = not any(are_scalars)

            # check if unambiguously given
            if not (all_scalars or all_tuples):
                raise ValueError("Has to be either a list of bounds or just the bounds (so everything"
                                 "a single value or tuples but not mixed). Is currently: {}".format(bounds))

            # sanitize, make 2-d
            if all_scalars:
                if dims is None:
                    raise ValueError("All bounds are scalars but dims is None -> ill-defined")
                # if len(bounds) == len(dims):  # only one limit
                bounds = (bounds,)
                # elif len(dims) == 1:  # several limits but only 1d
                #     bounds = tuple((b,) for b in bounds)

            # replace None
            for bound in bounds:
                if convert_none:
                    new_bounds.append(tuple(none_repl if b is None else b for b in bound))  # replace None
                else:
                    new_bounds.append(tuple(bound))
        inferred_dims = tuple(range(len(bound)))

        return tuple(new_lower), tuple(new_upper), inferred_dims

    @staticmethod
    def sanitize_limits(limits, dims=None, repl_none=False):
        are_scalars = [np.shape(l) == () for l in limits]
        all_scalars = all(are_scalars)
        all_tuples = not any(are_scalars)

        if not (all_scalars or all_tuples):
            raise ValueError("Invalid format for limits: {}".format(limits))

        if all_scalars:
            if len(limits) % 2 != 0:
                raise ValueError("Limits is 1-D but has an uneven number of entries. Ill-defined.")
            limits = (limits,)

        lower, upper = Range.boundaries_from_limits(limits=limits)
        *sanitized_boundaries, inferred_dims = Range.sanitize_boundaries(lower=lower, upper=upper, dims=dims,
                                                                         convert_none=repl_none)
        sanitized_limits = Range.limits_from_boundaries(*sanitized_boundaries)
        return sanitized_limits, inferred_dims

    @staticmethod
    def sanitize_dims(dims: Union[Tuple[int], int], allow_none=False) -> Tuple[int]:
        """Check the dims for dimensionality. None is error, Range.FULL is returned directly

        Args:
            allow_none ():
            dims (Union[Tuple[int], int]):

        Returns:
            Tuple[int]:
        """
        if dims is None and not allow_none:
            raise ValueError("`dims` cannot be None.")
        if dims is Range.FULL:
            return dims

        if len(np.shape(dims)) == 0:
            dims = (dims,)
        return dims

    def _set_boundaries_and_dims(self, lower, upper, dims, repl_none):
        # TODO all the conversions come here
        lower, upper, inferred_dims = self.sanitize_boundaries(lower=lower, upper=upper, dims=dims,
                                                               convert_none=repl_none)
        dims = self.sanitize_dims(dims, False)
        if dims is Range.FULL:
            dims = inferred_dims
        if dims is None:
            raise ValueError(
                "Due to safety: no dims provided but needed. Provide dims.")

        self._boundaries = lower, upper
        self._dims = tuple(dims)

    @property
    def n_dims(self):
        return len(self.dims)

    @property
    def area(self):
        """Return the total area of all the limits and dims. Useful, for example, for MC integration."""
        if self._area is None:
            self._calculate_save_area()
        return self._area

    def area_by_boundaries(self, rel=False):
        if self._area_by_boundaries is None:
            area_by_bound = [np.prod(np.array(up) - np.array(low)) for low, up in zip(*self.get_boundaries())]
        if rel:
            area_by_bound = np.array(area_by_bound) / self.area
        return tuple(area_by_bound)

    def _calculate_save_area(self):
        # area = 1.
        # for dims in self:
        #     sub_area = 0
        #     for lower, upper in iter_limits(dims):
        #         sub_area += upper - lower
        #     area *= sub_area
        area = sum(self.area_by_boundaries(rel=False))
        self._area = area
        return area

    @property
    def dims(self):
        return self._dims

    def get_limits(self) -> Tuple[Tuple[float, float]]:
        return Range.limits_from_boundaries(*self._boundaries)

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
        # print("DEBUG":, tuple", self.get_limits())
        lower, upper = Range.boundaries_from_limits(self.get_limits())
        return tuple(lower), tuple(upper)

    def subspace(self, dims: Tuple[int]) -> 'Range':
        """Return an instance of Range containing only a subspace (`dims`) of the instance"""
        dims = self.sanitize_dims(dims)
        lower, upper = self.get_boundaries()
        lower = tuple(tuple(lim[self.dims.index(d)] for d in dims) for lim in lower)
        upper = tuple(tuple(lim[self.dims.index(d)] for d in dims) for lim in upper)
        sub_range = Range(lower=lower, upper=upper, dims=dims)
        return sub_range

    def subbounds(self):
        """Return a list of Range objects each containing a simple boundary"""
        range_objects = []
        for lower, upper in zip(*self.get_boundaries()):
            range_objects.append(Range.from_boundaries(lower=lower, upper=upper, dims=self.dims))
        return range_objects

    @staticmethod
    def boundaries_from_limits(limits):
        if len(limits) == 0:
            return (), ()
        lower_limits = []
        upper_limits = []
        # iterate through the dimensions
        for lower, upper in iter_limits(limits[0]):  # recursive algorithm, append 0th dim element
            other_lower = None  # check if for loop gets executed
            for other_lower, other_upper in zip(*Range.boundaries_from_limits(limits[1:])):
                lower_limits.append(tuple([lower] + list(other_lower)))
                upper_limits.append(tuple([upper] + list(other_upper)))
            if other_lower is None:  # we're in the last axis
                lower_limits.append((lower,))
                upper_limits.append((upper,))
        return tuple(lower_limits), tuple(upper_limits)

    @staticmethod
    def limits_from_boundaries(lower, upper):
        lower = Range._add_dim_if_scalars(lower)
        upper = Range._add_dim_if_scalars(upper)

        if not np.shape(lower) == np.shape(upper) or len(np.shape(lower)) != 2:
            raise ValueError("lower {} and upper {} have to have the same (n_limits, n_dims) shape."
                             "Currently lower shape: {} upper shape: {}"
                             "".format(lower, upper, np.shape(lower), np.shape(upper)))
        limits = [[] for _ in range(len(lower[0]))]
        already_there_sets = [set() for _ in range(len(lower[0]))]
        for lower_vals, upper_vals in zip(lower, upper):
            for i, (lower_val, upper_val) in enumerate(zip(lower_vals, upper_vals)):
                new_limit = (lower_val, upper_val)
                if new_limit not in already_there_sets[i]:  # only extend if unique
                    limits[i].extend(new_limit)
                already_there_sets[i].add(new_limit)
        limits = tuple(tuple(limit) for limit in limits)

        # TODO: do some checks here?
        # TODO: sort somehow to make comparable
        check_lower, check_upper = Range.boundaries_from_limits(limits=limits)
        if not (set(check_lower) == set(lower) and set(check_upper) == set(upper)):
            raise ValueError("cannot safely convert boundaries (lower={}, upper={}) to limits "
                             "(check_lower={}, check_upper={}) (boundaries probably contain non "
                             "perpendicular limits)".format(lower, upper, check_lower, check_upper))
        # make boundaries unique
        return tuple(limits)

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
        raise ModuleNotFoundError("YEAH WRONG, but it's not implemented currently")
        # TODO: improve sorting for several Nones (how to sort?)
        if not any(obj is None for dim in limits for obj in dim):  # just a hack
            limits = tuple(tuple(sorted(list(vals))) for vals in limits)
        return tuple(Range._add_dim_if_scalars(limits))

    def __le__(self, other):  # TODO: refactor for boundaries
        if not isinstance(other, type(self)):
            raise TypeError("Comparison between other types than Range objects currently not "
                            "supported")
        if self.dims != other.dims:
            return False
        for dim, other_dim in zip(self.get_limits(), other.get_limits()):
            for lower, upper in iter_limits(dim):
                is_le = False
                for other_lower, other_upper in iter_limits(other_dim):
                    # a list of `or` conditions
                    is_le = other_lower == lower and upper == other_upper  # TODO: approx limit comparison?
                    is_le += other_lower == lower and other_upper is None  # TODO: approx limit comparison?
                    is_le += other_lower is None and upper == other_upper  # TODO: approx limit comparison?
                    is_le += other_lower is None and other_upper is None
                    if is_le:
                        break
                if not is_le:
                    return False
        return True

    def __ge__(self, other):
        return other.__le__(self)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Comparison between other types than Range objects currently not "
                            "supported")
        if self.dims != other.dims:
            return False
        own_lower, own_upper = self.get_boundaries()
        other_lower, other_upper = other.get_boundaries()
        lower_equal = set(own_lower) == set(other_lower)
        upper_equal = set(own_upper) == set(other_upper)
        are_equal = lower_equal and upper_equal

        return are_equal

    def __getitem__(self, key):
        raise Exception("Replace with .get_limits()")
        try:
            limits = tuple(self.get_limits()[axis] for axis in key)
        except TypeError:
            limits = self.get_limits()[key]
        return limits

    def idims_limits(self, dims):
        if not hasattr(dims, "__len__"):
            dims = (dims,)
        limits_by_dims = tuple([self.get_limits(self.dims.index(dim)) for dim in dims])
        return limits_by_dims

    def __hash__(self):
        try:
            return (self.get_boundaries(), self.__HASH_DELIMINATOR, self.dims).__hash__()
        except TypeError:
            raise TypeError("unhashable. ", self.get_boundaries(), self.dims)


def convert_to_range(limits=None, boundaries=None, dims=None, convert_none=False) -> Union[Range, bool, None]:
    """Convert *limits* to a Range object if not already None or False.

    Args:
        limits (Union[Tuple[float, float], zfit.core.limits.Range]):
        dims (Union[Range, False, None]):

    Returns:
        Union[Range, False, None]:
    """
    if limits is not None and boundaries is not None:
        raise ValueError("Both limits and boundaries are specified. Only use 1")
    if limits is None and boundaries is None:
        return None
    elif limits is False or boundaries is False:
        return False
    elif isinstance(limits, Range):
        return limits
    elif isinstance(boundaries, Range):
        return limits
    elif limits is not None:
        return Range.from_limits(limits=limits, dims=dims, convert_none=convert_none)
    elif boundaries is not None:
        lower, upper = boundaries
        return Range.from_boundaries(lower=lower, upper=upper, dims=dims, convert_none=convert_none)
    else:
        assert False, "This code block should never been reached."


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


def no_multiple_limits(func):
    """Decorator: Catch the 'limits' kwargs. If it contains multiple limits, raise MultipleLimitsNotImplementedError."""
    parameters = inspect.signature(func).parameters
    keys = list(parameters.keys())
    if 'limits' in keys:
        limits_index = keys.index('norm_range')
    else:
        raise TypeError("Decorator used to sanitize limits, but argument not given.")

    def new_func(*args, **kwargs):
        limits_is_arg = len(args) > limits_index
        if limits_is_arg:
            limits = args[limits_index]
        else:
            limits = kwargs['limits']

        if len(limits) > 1:
            raise MultipleLimitsNotImplementedError
        else:
            return func(*args, **kwargs)

    return new_func

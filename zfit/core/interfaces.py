#  Copyright (c) 2020 zfit

import abc
from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Callable, Tuple, Optional

import numpy as np
import tensorflow as tf

import zfit
from ..util import ztyping


class ZfitObject(abc.ABC):  # TODO(Mayou36): upgrade to tf2
    @property
    def name(self) -> str:
        """Name prepended to all ops created by this `model`."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def copy(self, deep: bool = False, **overwrite_params) -> "ZfitObject":
        raise NotImplementedError


class ZfitDimensional(ZfitObject):

    # @property
    # @abstractmethod
    # def space(self) -> "zfit.Space":
    #     """Return the :py:class:`~zfit.Space` object that defines the dimensionality of the object."""
    #     raise NotImplementedError

    @property
    @abstractmethod
    def obs(self) -> ztyping.ObsTypeReturn:
        """Return the observables."""
        raise NotImplementedError

    @property
    @abstractmethod
    def axes(self) -> ztyping.AxesTypeReturn:
        """Return the axes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Return the number of observables."""
        raise NotImplementedError


class ZfitOrderableDimensional(ZfitDimensional, metaclass=ABCMeta):

    @abstractmethod
    def with_obs(self, obs: Optional[ztyping.ObsTypeInput], allow_superset: bool = True,
                 allow_subset: bool = True):
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_axes(self, axes: Optional[ztyping.AxesTypeInput], allow_superset: bool = True,
                  allow_subset: bool = True):
        """Sort by `obs` and return the new instance.

        Args:
            axes ():

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abstractmethod
    def reorder_x(self, x, x_obs, x_axes, func_obs, func_axes):
        """Reorder x in the last dimension either according to its own obs or assuming a function ordered with func_obs.

        There are two obs or axes around: the one associated with this Coordinate object and the one associated with x.
        If x_obs or x_axes is given, then this is assumed to be the obs resp. the axes of x and x will be reordered
        according to `self.obs` resp. `self.axes`.

        If func_obs resp. func_axes is given, then x is assumed to have `self.obs` resp. `self.axes` and will be
        reordered to align with a function ordered with `func_obs` resp. `func_axes`.

        Switching `func_obs` for `x_obs` resp. `func_axes` for `x_axes` inverts the reordering of x.

        Args:
            x (tensor-like): Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, `func_obs` and `func_axes` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:

        """
        raise NotImplementedError


class ZfitData(ZfitDimensional):

    @abstractmethod
    def value(self, obs: List[str] = None) -> ztyping.XType:
        raise NotImplementedError

    @abstractmethod
    def sort_by_obs(self, obs, allow_superset: bool = True):
        raise NotImplementedError

    @abstractmethod
    def sort_by_axes(self, axes, allow_superset: bool = True):
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self):
        raise NotImplementedError


class ZfitLimit(abc.ABC, metaclass=ABCMeta):

    @property
    @abstractmethod
    def has_rect_limits(self) -> bool:
        """If there are limits and whether they are rectangular."""
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        """Return the rectangular limits as `np.ndarray``tf.Tensor` if they are set and not false.

            The rectangular limits can be used for sampling. They do not in general represent the limits
            of the object as a functional limit can be set and to check if something is inside the limits,
            the method :py:meth:`~Limit.inside` should be used.

            In order to test if the limits are False or None, it is recommended to use the appropriate methods
            `limits_are_false` and `limits_are_set`.

        Returns:
            tuple(np.ndarray/tf.Tensor, np.ndarray/tf.Tensor) or bool or None: The lower and upper limits.
        Raises:
            LimitsNotSpecifiedError: If there are not limits set or they are False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        """Return the rectangular limits as `np.ndarray`. Raise error if not possible.

        Rectangular limits are returned as numpy arrays which can be useful when doing checks that do not
        need to be involved in the computation later on as they allow direct interaction with Python as
        compared to `tf.Tensor` inside a graph function.

        In order to test if the limits are False or None, it is recommended to use the appropriate methods
        `limits_are_false` and `limits_are_set`.

        Returns:
            (lower, upper): A tuple of two `np.ndarray` with shape (1, n_obs) typically. The last
                dimension is always `n_obs`, the first can be vectorized. This allows unstacking
                with `z.unstack_x()` as can be done with data.

        Raises:
            CannotConvertToNumpyError: In case the conversion fails.
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        """The lower, rectangular limits, equivalent to `rect_limits[0]` with shape (..., n_obs)

        Returns:
            The lower, rectangular limits as `np.ndarray` or `tf.Tensor`
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_upper(self) -> ztyping.RectUpperReturnType:
        """The upper, rectangular limits, equivalent to `rect_limits[1]` with shape (..., n_obs)

        Returns:
            The upper, rectangular limits as `np.ndarray` or `tf.Tensor`
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @abstractmethod
    def rect_area(self) -> Union[float, np.ndarray, tf.Tensor]:
        """Calculate the total rectangular area of all the limits and axes. Useful, for example, for MC integration."""
        raise NotImplementedError

    @abstractmethod
    def inside(self, x: ztyping.XTypeInput, guarantee_limits: bool = False) -> ztyping.XTypeReturn:
        """Test if `x` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            tensor-like: Return a boolean tensor-like object with the same shape as the input `x` except of the
                last dimension removed.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, x: ztyping.XTypeInput,
               guarantee_limits: bool = False,
               axis: Optional[int] = None
               ) -> ztyping.XTypeReturnNoData:
        """Filter `x` by removing the elements along `axis` that are not inside the limits.

        This is similar to `tf.boolean_mask`.

        Args:
            x: Values to be checked whether they are inside of the limits. If not, the corresonding element (in the
                specified `axis`) is removed. The shape is expected to have the last dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.
            axis: The axis to remove the elements from. Defaults to 0.

        Returns:
            tensor-like: Return an object with the same shape as `x` except that along `axis` elements have been
                removed.
        """

    @property
    def rect_limits_are_tensors(self) -> bool:
        """Return True if the rectangular limits are tensors.

        If a limit with tensors is evaluated inside a graph context, comparison operations will fail.

        Returns:
            bool: if the rectangular limits are tensors.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_not_set(self) -> bool:
        """If the limits have not been set to a limit or to are False.

        Returns:
            bool:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_are_false(self) -> bool:
        """If the limits have been set to False, so the object on purpose does not contain limits.

        Returns:
            bool:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def has_limits(self) -> bool:
        """Whether there are limits set and they are not false.

        Returns:
            bool:
        """
        raise NotImplementedError

    # TODO: remove from API?
    def get_subspace(self, *_, **__):
        from zfit.util.exception import InvalidLimitSubspaceError
        raise InvalidLimitSubspaceError("ZfitLimits does not suppoert subspaces")

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Dimensionality, the number of observables, of the limits. Equals to the last axis in rectangular limits.

        Returns:
            int: Dimensionality of the limits.
        """
        raise NotImplementedError

    @property
    def n_events(self) -> Union[int, None]:
        """

        Returns:
            int, None: Return the number of events, the dimension of the first shape. If this is > 1 or None,
                it's vectorized.
        """
        raise NotImplementedError

    @abstractmethod
    def equal(self, other: object, allow_graph: bool) -> Union[bool, tf.Tensor]:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.

        Returns:
            bool: result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Returns:
            bool:
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def less_equal(self, other, allow_graph):
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            bool: result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            bool: result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sublimits(self):
        """Splits itself into multiple sublimits with smaller n_obs.

        If this is not possible, if the limits are not rectangular, just returns itself.

        Returns:
            Iterable[ZfitLimits]: The sublimits if it was able to split.
        """
        raise NotImplementedError


class ZfitSpace(ZfitLimit, ZfitOrderableDimensional, ZfitObject, metaclass=ABCMeta):

    @property
    @abstractmethod
    def obs(self) -> Tuple[str, ...]:
        """Return a list of the observable names.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_limits(self) -> int:
        """Return the number of limits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def axes(self) -> ztyping.AxesTypeReturn:
        raise NotImplementedError

    # TODO: remove below?
    # @abstractmethod
    # def get_axes(self, obs: Union[str, Tuple[str, ...]] = None, as_dict: bool = True):
    #     """Return the axes number of the observable *if available* (set by `axes_by_obs`).
    #
    #     Raises:
    #         AxesNotUnambiguousError: In case
    #     """
    #     raise NotImplementedError

    @property
    @abstractmethod
    def limits(self) -> Tuple[ztyping.LowerTypeReturn, ztyping.UpperTypeReturn]:
        """Return the tuple(lower, upper)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def upper(self) -> ztyping.UpperTypeReturn:
        """Return the upper limits.

        """
        raise NotImplementedError

    @abstractmethod
    def get_subspace(self, obs: ztyping.ObsTypeInput = None, axes=None, name=None) -> "zfit.Space":
        raise NotImplementedError

    @abstractmethod
    def area(self) -> float:
        """Return the total area of all the limits and axes. Useful, for example, for MC integration."""
        raise NotImplementedError

    # @abstractmethod
    # def iter_areas(self, rel: bool = False) -> Tuple[float, ...]:
    #     """Return the areas of each limit."""
    #     raise NotImplementedError

    @abstractmethod
    def with_limits(self, limits, name):
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits ():
            name (str):

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_obs(self, obs: Optional[ztyping.ObsTypeInput], allow_superset: bool = True,
                 allow_subset: bool = True):
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_axes(self, axes: Optional[ztyping.AxesTypeInput], allow_superset: bool = True,
                  allow_subset: bool = True):
        """Sort by `obs` and return the new instance.

        Args:
            axes ():

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_autofill_axes(self, overwrite: bool):
        """Return a :py:class:`~zfit.Space` with filled axes corresponding to range(len(n_obs)).

        Args:
            overwrite (bool): If `self.axes` is not None, replace the axes with the autofilled ones.
                If axes is already set, don't do anything if `overwrite` is False.

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abstractmethod
    def get_subspace(self, obs, axes, name):
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs (str, Tuple[str]):
            axes (int, Tuple[int]):
            name ():

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def with_coords(self, coords: ZfitOrderableDimensional, allow_superset: bool = True,
                    allow_subset: bool = True) -> object:
        """Return a new :py:class:`~zfit.Space` with reordered observables and set the `axes`.


        Args:
            coords (OrderedDict[str, int]): An ordered dict with {obs: axes}.
            ordered (bool): If True (and the `obs_axes` is an `OrderedDict`), the
            allow_subset ():

        Returns:
            :py:class:`~zfit.Space`:
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ZfitDependentsMixin:
    @abstractmethod
    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
        raise NotImplementedError


class ZfitNumeric(ZfitDependentsMixin, ZfitObject):
    @abstractmethod
    def get_params(self, only_floating: bool = False,
                   names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> tf.DType:
        """The `DType` of `Tensor`s handled by this `model`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> ztyping.ParametersType:
        raise NotImplementedError


class ZfitParameter(ZfitNumeric):

    @property
    @abstractmethod
    def floating(self) -> bool:
        raise NotImplementedError

    @floating.setter
    @abstractmethod
    def floating(self, value: bool):
        raise NotImplementedError

    @abstractmethod
    def value(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def independent(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> tf.DType:
        raise NotImplementedError

    @abc.abstractmethod
    def value(self):
        raise NotImplementedError


class ZfitLoss(ZfitObject, ZfitDependentsMixin, metaclass=ABCMeta):

    @abstractmethod
    def gradients(self, params: ztyping.ParamTypeInput = None) -> List[tf.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def value(self) -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @property
    @abstractmethod
    def errordef(self) -> Union[float, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> List["ZfitModel"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> List["ZfitData"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fit_range(self) -> List["ZfitSpace"]:
        raise NotImplementedError

    @abstractmethod
    def add_constraints(self, constraints: List[tf.Tensor]):
        raise NotImplementedError

    @property
    @abstractmethod
    def errordef(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def value_gradients(self, params):
        pass

    @abstractmethod
    def value_gradients_hessian(self, params, hessian=None):
        pass


class ZfitModel(ZfitNumeric, ZfitDimensional):

    @abstractmethod
    def update_integration_options(self, *args, **kwargs):  # TODO: handling integration properly
        raise NotImplementedError

    @abstractmethod
    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, :py:class:`~zfit.Space`): the limits to integrate over
            norm_range (tuple, :py:class:`~zfit.Space`): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str):

        Returns:
            Tensor: the integral value
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None,
                                   priority: int = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False):
        """Register an analytic integral with the class.

        Args:
            func ():
            limits (): |limits_arg_descr|
            priority (int):
            supports_multiple_limits (bool):
            supports_norm_range (bool):

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The value at which the partially integrated function will be evaluated
            limits (tuple, :py:class:`~zfit.Space`): the limits to integrate over. Can contain only some axes
            norm_range (tuple, :py:class:`~zfit.Space`, False): the limits to normalize over. Has to have all axes
            name (str):

        Returns:
            Tensor: the value of the partially integrated function evaluated at `x`.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def register_inverse_analytic_integral(cls, func: Callable):
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func ():
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, limits: ztyping.LimitsType = None, name: str = "sample") -> ztyping.XType:
        """Sample `n` points within `limits` from the model.

        Args:
            n (int): The number of samples to be generated
            limits (tuple, :py:class:`~zfit.Space`): In which region to sample in
            name (str):

        Returns:
            Tensor(n_obs, n_samples)
        """
        raise NotImplementedError


class ZfitFunc(ZfitModel):
    @abstractmethod
    def func(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        raise NotImplementedError

    @abstractmethod
    def as_pdf(self):
        raise NotImplementedError


class ZfitPDF(ZfitModel):

    @abstractmethod
    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, name: str = "model") -> ztyping.XType:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_extended(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_norm_range(self):
        raise NotImplementedError

    @abstractmethod
    def create_extended(self, yield_: ztyping.ParamTypeInput) -> "ZfitPDF":
        raise NotImplementedError

    @abstractmethod
    def get_yield(self) -> Union[ZfitParameter, None]:
        raise NotImplementedError

    @abstractmethod
    def normalization(self, limits: ztyping.LimitsType, name: str = "normalization") -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @abstractmethod
    def as_func(self, norm_range: ztyping.LimitsType = False):
        raise NotImplementedError


class ZfitFunctorMixin:

    @property
    @abstractmethod
    def models(self) -> Dict[Union[float, int, str], ZfitModel]:
        raise NotImplementedError

    @abstractmethod
    def get_models(self) -> List[ZfitModel]:
        raise NotImplementedError


class ZfitConstraint(abc.ABC):
    @abstractmethod
    def value(self):
        raise NotImplementedError

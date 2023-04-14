#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

import abc
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from uhi.typing.plottable import PlottableHistogram

from ..util import ztyping
from ..util.deprecation import deprecated


class ZfitObject(abc.ABC):
    # TODO: make abstractmethod?
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError


class ZfitDimensional(ZfitObject):
    @property
    @abstractmethod
    def obs(self) -> ztyping.ObsTypeReturn:
        """Return the observables, string identifier for the coordinate system."""
        raise NotImplementedError

    @property
    @abstractmethod
    def axes(self) -> ztyping.AxesTypeReturn:
        """Return the axes, integer based identifier(indices) for the coordinate system."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Return the number of observables, the dimensionality.

        Corresponds to the last dimension.
        """
        raise NotImplementedError

    # TODO: activate?
    # @property
    # @abstractmethod
    # def space(self):
    #     raise NotImplementedError


class ZfitOrderableDimensional(ZfitDimensional, metaclass=ABCMeta):
    @abstractmethod
    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitOrderableDimensional:
        """Create a new instance that has ``obs``; sorted by or set or dropped.

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
            ObsIncompatibleError: if ``obs`` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        raise NotImplementedError

    @abstractmethod
    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitOrderableDimensional:
        """Create a new instance that has ``axes``; sorted by or set or dropped.

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
            AxesIncompatibleError: if ``axes`` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """

        raise NotImplementedError

    @abstractmethod
    def with_autofill_axes(self, overwrite: bool = False) -> ZfitOrderableDimensional:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. ``overwrite`` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and ``overwrite`` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and ``overwrite`` is False.
        """
        raise NotImplementedError

    @abstractmethod
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
        according to ``self.obs`` resp. ``self.axes``.

        If func_obs resp. func_axes is given, then x is assumed to have ``self.obs`` resp. ``self.axes`` and will be
        reordered to align with a function ordered with ``func_obs`` resp. ``func_axes``.

        Switching ``func_obs`` for ``x_obs`` resp. ``func_axes`` for ``x_axes`` inverts the reordering of x.

        Args:
            x: Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, ``func_obs`` and ``func_axes`` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:
            The reordered array-like object
        """
        raise NotImplementedError

    @abstractmethod
    def get_reorder_indices(
        self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None
    ) -> tuple[int]:
        """Indices that would order the instances obs as ``obs`` respectively the instances axes as ``axes``.

        Args:
            obs: Observables that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.
            axes: Axes that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.

        Returns:
            New indices that would reorder the instances obs to be obs respectively axes.

        Raises:
            CoordinatesUnderdefinedError: If neither ``obs`` nor ``axes`` is given
        """
        raise NotImplementedError


class ZfitData(ZfitDimensional):
    @abstractmethod
    def value(self, obs: list[str] = None) -> ztyping.XType:
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


class ZfitUnbinnedData(ZfitData):
    pass


class ZfitLimit(abc.ABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        """Return the rectangular limits as ``np.ndarray``tf.Tensor`` if they are set and not false.

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
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        """Return the rectangular limits as ``np.ndarray``. Raise error if not possible.

        Rectangular limits are returned as numpy arrays which can be useful when doing checks that do not
        need to be involved in the computation later on as they allow direct interaction with Python as
        compared to ``tf.Tensor`` inside a graph function.

        In order to test if the limits are False or None, it is recommended to use the appropriate methods
        ``limits_are_false`` and ``limits_are_set``.

        Returns:
            A tuple of two ``np.ndarray`` with shape (1, n_obs) typically. The last
                dimension is always ``n_obs``, the first can be vectorized. This allows unstacking
                with ``z.unstack_x()`` as can be done with data.

        Raises:
            CannotConvertToNumpyError: In case the conversion fails.
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        """The lower, rectangular limits, equivalent to ``rect_limits[0]`` with shape (..., n_obs)

        Returns:
            The lower, rectangular limits as ``np.ndarray`` or ``tf.Tensor``
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rect_upper(self) -> ztyping.RectUpperReturnType:
        """The upper, rectangular limits, equivalent to ``rect_limits[1]`` with shape (..., n_obs)

        Returns:
            The upper, rectangular limits as ``np.ndarray`` or ``tf.Tensor``
        Raises:
            LimitsNotSpecifiedError: If the limits are not set or are false
        """
        raise NotImplementedError

    @abstractmethod
    def rect_area(self) -> float | np.ndarray | tf.Tensor:
        """Calculate the total rectangular area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        raise NotImplementedError

    @abstractmethod
    def inside(
        self, x: ztyping.XTypeInput, guarantee_limits: bool = False
    ) -> ztyping.XTypeReturn:
        """Test if ``x`` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            Return a boolean tensor-like object with the same shape as the input ``x`` except of the
                last dimension removed.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        x: ztyping.XTypeInput,
        guarantee_limits: bool = False,
        axis: int | None = None,
    ) -> ztyping.XTypeReturnNoData:
        """Filter ``x`` by removing the elements along ``axis`` that are not inside the limits.

        This is similar to ``tf.boolean_mask``.

        Args:
            x: Values to be checked whether they are inside of the limits. If not, the corresonding element (in the
                specified ``axis``) is removed. The shape is expected to have the last dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.
            axis: The axis to remove the elements from. Defaults to 0.

        Returns:
            Return an object with the same shape as ``x`` except that along ``axis`` elements have been
                removed.
        """

    @property
    @abstractmethod
    def has_rect_limits(self) -> bool:
        """If there are limits and whether they are rectangular."""
        raise NotImplementedError

    @property
    def rect_limits_are_tensors(self) -> bool:
        """Return True if the rectangular limits are tensors.

        If a limit with tensors is evaluated inside a graph context, comparison operations will fail.

        Returns:
            If the rectangular limits are tensors.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_are_set(self) -> bool:
        """If the limits have been set to a limit or are False.

        Returns:
            Whether the limits have been set or not.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_are_false(self) -> bool:
        """Returns if the limits have been set to False, so the object on purpose does not contain limits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_limits(self) -> bool:
        """Whether there are limits set and they are not false."""
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
            Dimensionality of the limits.
        """
        raise NotImplementedError

    @property
    def n_events(self) -> int | None:
        """Shape of the first dimension, usually reflects the number of events.

        Returns:
            Return the number of events, the dimension of the first shape. If this is > 1 or None,
                it's vectorized.
        """
        raise NotImplementedError

    @abstractmethod
    def equal(self, other: object, allow_graph: bool) -> bool | tf.Tensor:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic ``tf.Tensor``.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def less_equal(self, other: object, allow_graph: bool = True) -> bool | tf.Tensor:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic ``tf.Tensor``.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sublimits(self):
        """Splits itself into multiple sublimits with smaller n_obs.

        If this is not possible, if the limits are not rectangular, just returns itself.

        Returns:
            The sublimits if it was able to split.
        """
        raise NotImplementedError


class ZfitSpace(ZfitLimit, ZfitOrderableDimensional, ZfitObject, metaclass=ABCMeta):
    @property
    def is_binned(self):
        raise NotImplementedError

    @property
    def binning(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_limits(self) -> int:
        """Return the number of limits."""
        raise NotImplementedError

    # TODO: legacy?
    @property
    @abstractmethod
    def limits(self) -> tuple[ztyping.LowerTypeReturn, ztyping.UpperTypeReturn]:
        """Return the tuple(lower, upper)."""
        raise NotImplementedError

    # TODO: legacy?
    @property
    @abstractmethod
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits."""
        raise NotImplementedError

    # TODO: legacy?
    @property
    @abstractmethod
    def upper(self) -> ztyping.UpperTypeReturn:
        """Return the upper limits."""
        raise NotImplementedError

    # TODO: legacy?
    @abstractmethod
    def area(self) -> float:
        """Return the total area of all the limits and axes.

        Useful, for example, for MC integration.
        """
        raise NotImplementedError

    @abstractmethod
    def with_limits(
        self,
        limits: ztyping.LimitsTypeInput = None,
        rect_limits: ztyping.RectLimitsInputType | None = None,
        name: str | None = None,
    ) -> ZfitSpace:
        """Return a copy of the space with the new ``limits`` (and the new ``name``).

        Args:
            limits: Limits to use. Can be rectangular, a function (requires to also specify ``rect_limits``
                or an instance of ZfitLimit.
            rect_limits: Rectangular limits that will be assigned with the instance
            name: Human readable name

        Returns:
            Copy of the current object with the new limits.
        """
        raise NotImplementedError

    def with_binning(self, binning: ztyping.BinningTypeInput) -> ZfitSpace:
        """Return a copy of the space with the new ``binning``.

        Args:
            binning: Binning to use.

        Returns:
            Copy of the current object with the new binning.
        """
        raise NotImplementedError

    @abstractmethod
    def get_subspace(self, obs, axes, name):
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs:
            axes:
            name:

        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def with_coords(
        self,
        coords: ZfitOrderableDimensional,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> object:
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
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ZfitDependenciesMixin:
    @abstractmethod
    def get_cache_deps(self, only_floating: bool = True) -> ztyping.DependentsType:
        raise NotImplementedError

    @deprecated(
        date=None,
        instructions="Use `get_params` instead if you want to retrieve the "
        "independent parameters or `get_cache_deps` in case you need "
        "the numerical cache dependents (advanced).",
    )
    def get_dependencies(self, only_floating: bool = True) -> ztyping.DependentsType:
        # raise BreakingAPIChangeError
        return self.get_cache_deps(only_floating=only_floating)


class ZfitParametrized(ZfitDependenciesMixin, ZfitObject):
    @abstractmethod
    def get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
    ) -> set[ZfitParameter]:
        """Recursively collect parameters that this object depends on according to the filter criteria.

        Which parameters should be included can be steered using the arguments as a filter.
         - **None**: do not filter on this. E.g. `floating=None` will return parameters that are floating as well as
            parameters that are fixed.
         - **True**: only return parameters that fulfil this criterion
         - **False**: only return parameters that do not fulfil this criterion. E.g. `floating=False` will return
            only parameters that are not floating.

        Args:
            floating: if a parameter is floating, e.g. if :py:meth:`~ZfitParameter.floating` returns `True`
            is_yield: if a parameter is a yield of the _current_ model. This won't be applied recursively, but may include
               yields if they do also represent a parameter parametrizing the shape. So if the yield of the current
               model depends on other yields (or also non-yields), this will be included. If, however, just submodels
               depend on a yield (as their yield) and it is not correlated to the output of our model, they won't be
               included.
            extract_independent: If the parameter is an independent parameter, i.e. if it is a `ZfitIndependentParameter`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> ztyping.ParameterType:
        raise NotImplementedError


class ZfitNumericParametrized(ZfitParametrized):
    @property
    @abstractmethod
    def dtype(self) -> tf.DType:
        """The `DType` of `Tensor`s handled by this `model`."""
        raise NotImplementedError


class ZfitParameter(ZfitNumericParametrized):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    # TODO: maybe add to numerics?
    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

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

    @abstractmethod
    def read_value(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def independent(self) -> bool:
        raise NotImplementedError


class ZfitIndependentParameter(ZfitParameter, metaclass=ABCMeta):
    @abstractmethod
    def randomize(self, minval, maxval, sampler):
        """Update the parameter with a randomised value between minval and maxval and return it.

        Args:
            minval: The lower bound of the sampler. If not given, `lower_limit` is used.
            maxval: The upper bound of the sampler. If not given, `upper_limit` is used.
            sampler: A sampler with the same interface as `tf.random.uniform`

        Returns:
            The sampled value
        """
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value):
        """Set the :py:class:`~zfit.Parameter` to `value` (temporarily if used in a context manager).

        This operation won't, compared to the assign, return the read value but an object that *can* act as a context
        manager.

        Args:
            value: The value the parameter will take on.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def has_limits(self) -> bool:
        """If the parameter has limits set or not."""
        raise NotImplementedError

    @property
    @abstractmethod
    def at_limit(self) -> tf.Tensor:
        """If the value is at the limit (or over it).

        Returns:
            Boolean `tf.Tensor` that tells whether the value is at the limits.
        """
        raise NotImplementedError

    @property
    def step_size(self) -> tf.Tensor:
        """Step size of the parameter, the estimated order of magnitude of the uncertainty.

        This can be crucial to tune for the minimization. A too large `step_size` can produce NaNs, a too small won't
        converge.

        If the step size is not set, the `DEFAULT_STEP_SIZE` is used.

        Returns:
            The step size
        """
        raise NotImplementedError


class ZfitLoss(ZfitObject, metaclass=ABCMeta):
    @abstractmethod
    def gradient(self, params: ztyping.ParamTypeInput = None) -> list[tf.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def value(self) -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> list[ZfitModel]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> list[ZfitData]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fit_range(self) -> list[ZfitSpace]:
        raise NotImplementedError

    @abstractmethod
    def add_constraints(self, constraints: list[tf.Tensor]):
        raise NotImplementedError

    @property
    @abstractmethod
    def errordef(self) -> float:
        raise NotImplementedError

    def hessian(self, params):
        pass

    @abstractmethod
    def value_gradient(self, params):
        pass

    @abstractmethod
    def value_gradient_hessian(self, params, hessian=None):
        pass

    @abstractmethod
    def create_new(self, **kwargs):
        pass


class ZfitModel(ZfitNumericParametrized, ZfitDimensional):
    @abstractmethod
    def update_integration_options(
        self, *args, **kwargs
    ):  # TODO: handling integration properly
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self, limits: ztyping.LimitsType, norm: ztyping.LimitsType = None, *, options
    ) -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            * ():
            options ():
            limits: the limits to integrate over
            norm: the limits to normalize over or False to integrate the
                unnormalized probability
            name:

        Returns:
            The integral value
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def register_analytic_integral(
        cls,
        func: Callable,
        limits: ztyping.LimitsType = None,
        priority: int = 50,
        *,
        supports_norm: bool = False,
        supports_multiple_limits: bool = False,
    ):
        """Register an analytic integral with the class.

        Args:
            func:
            limits: |limits_arg_descr|
            priority:
            supports_multiple_limits:
            supports_norm:

        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def partial_integrate(
        self,
        x: ztyping.XType,
        limits: ztyping.LimitsType,
        *,
        norm=None,
        options=None,
        norm_range: ztyping.LimitsType = None,
    ) -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            * ():
            norm ():
            options ():
            x: The value at which the partially integrated function will be evaluated
            limits: the limits to integrate over. Can contain only some axes
            norm_range: the limits to normalize over. Has to have all axes

        Returns:
            The value of the partially integrated function evaluated at `x`.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def register_inverse_analytic_integral(cls, func: Callable):
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func:
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, limits: ztyping.LimitsType = None) -> ztyping.XType:
        """Sample `n` points within `limits` from the model.

        Args:
            n: The number of samples to be generated
            limits: In which region to sample in
            name:

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
    def pdf(
        self, x: ztyping.XType, norm: ztyping.LimitsType = None, norm_range=None
    ) -> ztyping.XType:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_extended(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_norm_range(self):
        raise NotImplementedError

    @abstractmethod
    def create_extended(
        self, yield_: ztyping.ParamTypeInput, name: str = None
    ) -> ZfitPDF:
        raise NotImplementedError

    @abstractmethod
    def get_yield(self) -> ZfitParameter | None:
        raise NotImplementedError

    @abstractmethod
    def normalization(
        self, limits: ztyping.LimitsType, *, options
    ) -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @abstractmethod
    def as_func(self, norm_range: ztyping.LimitsType = False):
        raise NotImplementedError


class ZfitFunctorMixin:
    @property
    @abstractmethod
    def models(self) -> dict[float | int | str, ZfitModel]:
        raise NotImplementedError

    @abstractmethod
    def get_models(self) -> list[ZfitModel]:
        raise NotImplementedError


class ZfitConstraint(abc.ABC):
    @abstractmethod
    def value(self):
        raise NotImplementedError


class ZfitMinimalHist(PlottableHistogram):
    @property
    def kind(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def variances(self) -> np.ndarray | None:
        raise NotImplementedError

    def axes(self):
        raise NotImplementedError


class ZfitBinnedData(ZfitDimensional, ZfitMinimalHist, metaclass=ABCMeta):
    @abstractmethod
    def variances(self):
        raise NotImplementedError

    def with_obs(self, obs) -> ZfitBinnedData:
        raise NotImplementedError

    # @abstractmethod
    # def counts(self):  # TODO: name?
    #     raise NotImplementedError

    # @abstractmethod
    # def binning(self):
    #     return self.space.binning
    @abstractmethod
    def to_hist(self):
        """Convert the binned data to a :py:class:`~hist.NamedHist`.

        While a binned data object can be used inside zfit (PDFs,...), it lacks many convenience features that the
        `hist library <https://hist.readthedocs.io/>`_
        offers, such as plots.
        """
        pass


class ZfitBinnedPDF(ZfitPDF, metaclass=ABCMeta):
    @abstractmethod
    def counts(self, x, norm):
        pass

    @abstractmethod
    def rel_counts(self, x, norm):
        pass


class ZfitBinning:
    pass


class ZfitRectBinning(ZfitBinning):
    @abstractmethod
    def get_edges(self):
        raise NotImplementedError

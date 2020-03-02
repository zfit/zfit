#  Copyright (c) 2020 zfit

import abc
from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Callable, Tuple, Optional

import tensorflow as tf

import zfit
from ..util import ztyping
from ..util.exception import BreakingAPIChangeError


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
    def with_obs(self, obs: Optional[ztyping.ObsTypeInput], allow_superset: bool = False,
                 allow_subset: bool = False):
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_axes(self, axes: Optional[ztyping.AxesTypeInput], allow_superset: bool = False,
                  allow_subset: bool = False):
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
    def sort_by_obs(self, obs, allow_superset: bool = False):
        raise NotImplementedError

    @abstractmethod
    def sort_by_axes(self, axes, allow_superset: bool = False):
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self):
        raise NotImplementedError


class ZfitLimit(abc.ABC):

    @property
    # @abstractmethod  # TODO(spaces): make abstract
    def rect_limits(self):
        raise NotImplementedError

    @property
    # @abstractmethod  # TODO(spaces): make abstract
    def rect_lower(self):
        raise NotImplementedError

    @property
    # @abstractmethod  # TODO(spaces): make abstract
    def rect_upper(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def has_rect_limits(self) -> bool:
        """If the limits are rectangular."""
        raise NotImplementedError

    @abstractmethod
    def rect_area(self) -> float:
        """Return the total rectangular area of all the limits and axes. Useful, for example, for MC integration."""
        raise NotImplementedError

    @abstractmethod
    def inside(self, x, guarantee_limits):
        raise NotImplementedError

    @abstractmethod
    def filter(self, x, guarantee_limits):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Return the number of observables (axis)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_not_set(self):
        raise NotImplementedError

    @property
    def rect_limits_are_tensors(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def has_limits(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def limits_are_false(self):
        raise NotImplementedError

    @property
    # @abstractmethod  # TODO(spaces): make abstact
    def limit_fn(self):
        raise NotImplementedError

    def get_subspace(self, *_, **__):
        from zfit.util.exception import InvalidLimitSubspaceError
        raise InvalidLimitSubspaceError("ZfitLimits does not suppoert subspaces")

    # @abstractmethod  # TODO(spaces)
    def equal(self, other, allow_graph):
        pass


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
    def with_obs(self, obs: Optional[ztyping.ObsTypeInput], allow_superset: bool = False,
                 allow_subset: bool = False):
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `Space`
        """
        raise NotImplementedError

    @abstractmethod
    def with_axes(self, axes: Optional[ztyping.AxesTypeInput], allow_superset: bool = False,
                  allow_subset: bool = False):
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

    # @classmethod
    # @abstractmethod
    # def from_axes(cls, axes, limits, name):
    #     """Create a space from `axes` instead of from `obs`.
    #
    #     Args:
    #         axes ():
    #         limits ():
    #         name (str):
    #
    #     Returns:
    #         :py:class:`~zfit.Space`
    #     """
    #     pass

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
    def with_coords(self, coords, allow_superset=False, allow_subset=True):
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

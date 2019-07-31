#  Copyright (c) 2019 zfit

import abc
from collections import OrderedDict
from typing import Union, List, Dict, Callable, Tuple, Mapping, Iterable

import tensorflow as tf

import zfit
from ..util import ztyping


class ZfitObject(metaclass=abc.ABCMeta):
    # class ZfitObject:
    @property
    # @abc.abstractmethod
    def name(self) -> str:
        """Name prepended to all ops created by this `model`."""
        raise NotImplementedError

    # @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    # @abc.abstractmethod
    def copy(self, deep: bool = False, **overwrite_params) -> "ZfitObject":
        raise NotImplementedError


class ZfitDimensional(ZfitObject):

    @property
    @abc.abstractmethod
    def space(self) -> "zfit.Space":
        """Return the :py:class:`~zfit.Space` object that defines the dimensionality of the object."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs(self) -> ztyping.ObsTypeReturn:
        """Return the observables."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def axes(self) -> ztyping.AxesTypeReturn:
        """Return the axes."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_obs(self) -> int:
        """Return the number of observables."""
        raise NotImplementedError


class ZfitData(ZfitDimensional):

    @abc.abstractmethod
    def value(self, obs: List[str] = None) -> ztyping.XType:
        raise NotImplementedError

    @abc.abstractmethod
    def sort_by_obs(self, obs, allow_superset: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def sort_by_axes(self, axes, allow_superset: bool = False):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def weights(self):
        raise NotImplementedError


class ZfitSpace(ZfitObject):

    @property
    @abc.abstractmethod
    def obs(self) -> Tuple[str, ...]:
        """Return a list of the observable names.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_limits(self) -> int:
        """Return the number of limits."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_obs(self) -> int:
        """Return the number of observables (axis)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def axes(self) -> ztyping.AxesTypeReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def get_axes(self, obs: Union[str, Tuple[str, ...]] = None, as_dict: bool = True):
        """Return the axes number of the observable *if available* (set by `axes_by_obs`).

        Raises:
            AxesNotUnambiguousError: In case
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def limits(self) -> Tuple[ztyping.LowerTypeReturn, ztyping.UpperTypeReturn]:
        """Return the tuple(lower, upper)."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_limits(self):
        """Iterate through the limits by returning several observables/(lower, upper)-tuples.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def upper(self) -> ztyping.UpperTypeReturn:
        """Return the upper limits.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_subspace(self, obs: ztyping.ObsTypeInput = None, axes=None, name=None) -> "zfit.Space":
        raise NotImplementedError

    @abc.abstractmethod
    def area(self) -> float:
        """Return the total area of all the limits and axes. Useful, for example, for MC integration."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_areas(self, rel: bool = False) -> Tuple[float, ...]:
        """Return the areas of each limit."""
        raise NotImplementedError

    @abc.abstractmethod
    def with_limits(self, limits, name):
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits ():
            name (str):

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def with_obs(self, obs):
        """Sort by `obs` and return the new instance.

        Args:
            obs ():

        Returns:
            `Space`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def with_axes(self, axes):
        """Sort by `obs` and return the new instance.

        Args:
            axes ():

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def with_autofill_axes(self, overwrite: bool):
        """Return a :py:class:`~zfit.Space` with filled axes corresponding to range(len(n_obs)).

        Args:
            overwrite (bool): If `self.axes` is not None, replace the axes with the autofilled ones.
                If axes is already set, don't do anything if `overwrite` is False.

        Returns:
            :py:class:`~zfit.Space`
        """
        raise NotImplementedError


class ZfitDependentsMixin:
    @abc.abstractmethod
    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
        raise NotImplementedError


class ZfitNumeric(ZfitDependentsMixin, ZfitObject):
    @abc.abstractmethod
    def get_params(self, only_floating: bool = False,
                   names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> tf.DType:
        """The `DType` of `Tensor`s handled by this `model`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self) -> ztyping.ParametersType:
        raise NotImplementedError


class ZfitParameter(ZfitNumeric):

    @property
    @abc.abstractmethod
    def floating(self) -> bool:
        raise NotImplementedError

    @floating.setter
    @abc.abstractmethod
    def floating(self, value: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def value(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def independent(self) -> bool:
        raise NotImplementedError


class ZfitLoss(ZfitObject, ZfitDependentsMixin):

    @abc.abstractmethod
    def gradients(self, params: ztyping.ParamTypeInput = None) -> List[tf.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def value(self) -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def errordef(self) -> Union[float, int]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model(self) -> List["ZfitModel"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data(self) -> List["ZfitData"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fit_range(self) -> List["ZfitSpace"]:
        raise NotImplementedError

    @abc.abstractmethod
    def add_constraints(self, constraints: List[tf.Tensor]):
        raise NotImplementedError


class ZfitModel(ZfitNumeric, ZfitDimensional):

    @abc.abstractmethod
    def update_integration_options(self, *args, **kwargs):  # TODO: handling integration properly
        raise NotImplementedError

    @abc.abstractmethod
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
    @abc.abstractmethod
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

    @abc.abstractmethod
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
    @abc.abstractmethod
    def register_inverse_analytic_integral(cls, func: Callable):
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func ():
        """
        raise NotImplementedError

    @abc.abstractmethod
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
    @abc.abstractmethod
    def func(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        raise NotImplementedError

    @abc.abstractmethod
    def as_pdf(self):
        raise NotImplementedError


class ZfitPDF(ZfitModel):

    @abc.abstractmethod
    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, name: str = "model") -> ztyping.XType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_extended(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def set_norm_range(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_extended(self, yield_: ztyping.ParamTypeInput) -> "ZfitPDF":
        raise NotImplementedError

    @abc.abstractmethod
    def get_yield(self) -> Union[ZfitParameter, None]:
        raise NotImplementedError

    @abc.abstractmethod
    def normalization(self, limits: ztyping.LimitsType, name: str = "normalization") -> ztyping.NumericalTypeReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def as_func(self, norm_range: ztyping.LimitsType = False):
        raise NotImplementedError


class ZfitFunctorMixin:

    @property
    @abc.abstractmethod
    def models(self) -> Dict[Union[float, int, str], ZfitModel]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_models(self) -> List[ZfitModel]:
        raise NotImplementedError

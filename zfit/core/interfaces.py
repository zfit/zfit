import abc
from collections import OrderedDict
import typing
from typing import Union, List, Dict, Callable, Tuple, Mapping, Iterable

import pep487
import tensorflow as tf

import zfit
from ..util import ztyping


class ZfitObject(pep487.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name prepended to all ops created by this `model`."""
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self, deep: bool = False, **overwrite_params) -> "ZfitObject":
        raise NotImplementedError

    @abc.abstractmethod
    def _repr(self):  # TODO: needed? Should fully represent the object
        raise NotImplementedError


class ZfitData(ZfitObject):
    @abc.abstractmethod
    def value(self, names: str = None) -> ztyping.XType:
        raise NotImplementedError

    @property
    def space(self) -> "ZfitObservable":
        raise NotImplementedError

    @space.setter
    def space(self, value: ztyping.InputObservableType):
        raise NotImplementedError


class ZfitObservable(ZfitObject):

    @property
    @abc.abstractmethod
    def names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_subspace(self, names: str = None) -> "ZfitObservable":
        raise NotImplementedError

    @abc.abstractmethod
    def iter_limits(self, as_tuple: bool = True):
        """Iterate through the limits by returning several observables/(lower, upper)-tuples.

        Args:
            as_tuple (bool): If True, return the (lower, upper) tuples instead of several
                observables.

        """
        raise NotImplementedError


class ZfitNamedSpace(ZfitObject):

    @property
    @abc.abstractmethod
    def obs(self) -> Tuple[str, ...]:
        """Return a list of the observable names.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_axes(self, obs: Union[str, Tuple[str, ...]] = None, as_dict: bool = True):
        """Return the axes number of the observable *if available* (set by `axes_by_obs`).

        Raises:
            AxesNotUnambiguousError: In case
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_subspace(self, obs: ztyping.InputObservableType = None, axes=None, name=None) -> "ZfitNamedSpace":
        raise NotImplementedError

    @abc.abstractmethod
    def iter_limits(self):
        """Iterate through the limits by returning several observables/(lower, upper)-tuples.

        """
        raise NotImplementedError

    def iter_space(self) -> List["ZfitNamedSpace"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def limits(self) -> Tuple[ztyping.ReturnLowerType, ztyping.ReturnUpperType]:
        """Return the tuple(lower, upper)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def lower(self) -> ztyping.ReturnLowerType:
        """Return the lower limits.

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def upper(self) -> ztyping.ReturnUpperType:
        """Return the upper limits.

        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def get_limits(self):

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

    @abc.abstractmethod
    def set_limits(self, lower: ztyping.InputLowerType, upper: ztyping.InputUpperType):
        """Set the limits of the NamedSpace (temporarily)."""
        raise NotImplementedError

    @abc.abstractmethod
    def area(self) -> float:
        """Return the total area of all the limits and axes. Useful, for example, for MC integration."""
        raise NotImplementedError

    @abc.abstractmethod
    def iter_areas(self, rel: bool = False) -> Tuple[float, ...]:
        """Return the areas of each limit."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def axes(self) -> List[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs_axes(self, autofill: bool = False) -> typing.Dict[str, int]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_obs_axes(self, obs_axes: ztyping.OrderedDict[str, int]):  # TODO: switch if sorting?
        raise NotImplementedError


class ZfitDependentsMixin(pep487.ABC):
    @abc.abstractmethod
    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
        raise NotImplementedError


class ZfitNumeric(ZfitObject, ZfitDependentsMixin):
    @abc.abstractmethod
    def get_parameters(self, only_floating: bool = False,
                       names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> tf.DType:
        """The `DType` of `Tensor`s handled by this `model`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self) -> ztyping.ParametersType:
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
    def value(self) -> ztyping.ReturnNumericalType:
        raise NotImplementedError

    @abc.abstractmethod
    def errordef(self, sigma: Union[float, int]) -> Union[float, int]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model(self) -> List["ZfitModel"]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data(self) -> "TODO(mayou36): add Dataset":
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fit_range(self) -> List["zfit.Range"]:
        raise NotImplementedError

    @abc.abstractmethod
    def add_constraints(self, constraints: Dict[ZfitParameter, "ZfitModel"]):
        raise NotImplementedError


class ZfitModel(ZfitNumeric):

    @property
    @abc.abstractmethod
    def n_dims(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dims(self) -> ztyping.AxesType:
        raise NotImplementedError

    @dims.setter
    @abc.abstractmethod
    def dims(self, value: ztyping.AxesType):
        raise NotImplementedError

    @abc.abstractmethod
    def set_integration_options(self, mc_options: dict = None, numeric_options: dict = None,
                                general_options: dict = None, analytic_options: dict = None):
        raise NotImplementedError

    @abc.abstractmethod
    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str):

        Returns:
            Tensor: the integral value
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None,
                                   dims: ztyping.AxesType = None, priority: int = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False):
        """Register an analytic integral with the class.

        Args:
            func ():
            limits (): |limits_arg_descr|
            dims (tuple(int)):
            priority (int):
            supports_multiple_limits (bool):
            supports_norm_range (bool):

        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.AxesType = None,
                          norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some axes
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all axes
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
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
    def sample(self, n: int, limits: ztyping.LimitsType, name: str = "sample") -> ztyping.XType:
        """Sample `n` points within `limits` from the model.

        Args:
            n (int): The number of samples to be generated
            limits (tuple, Range): In which region to sample in
            name (str):

        Returns:
            Tensor(n_dims, n_samples)
        """
        raise NotImplementedError

    @property
    def obs(self) -> Tuple["Observable"]:
        raise NotImplementedError

    @obs.setter
    def obs(self, value: ztyping.InputObservableType):
        raise NotImplementedError


class ZfitFunc(ZfitModel):
    @abc.abstractmethod
    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
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
    def set_yield(self, value: Union[ZfitParameter, None]):
        raise NotImplementedError

    @abc.abstractmethod
    def get_yield(self) -> Union[ZfitParameter, None]:
        raise NotImplementedError

    @abc.abstractmethod
    def normalization(self) -> ztyping.ReturnNumericalType:
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

    # @property
    # @abc.abstractmethod
    # def axes(self):
    #     raise NotImplementedError

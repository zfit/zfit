import abc
from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Callable

import pep487
import tensorflow as tf

import zfit
from zfit.util.ztyping import ReturnNumericalType
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
    def value(self) -> ReturnNumericalType:
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
    @abstractmethod
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
    @abstractmethod
    def register_analytic_integral(cls, func: Callable, limits: ztyping.LimitsType = None,
                                   dims: ztyping.DimsType = None, priority: int = 50, *,
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

    @classmethod
    @abstractmethod
    def register_inverse_analytic_integral(cls, func: Callable):
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func ():
        """
        raise NotImplementedError

    @abstractmethod
    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType = None,
                          norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
        """
        raise NotImplementedError

    @abstractmethod
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


class ZfitFunc(ZfitModel):
    @abc.abstractmethod
    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
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

import abc
from abc import ABCMeta, abstractmethod
from typing import Union, List

import pep487
import tensorflow as tf

from ..util import ztyping


class ZfitObject(pep487.ABC, metaclass=ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name prepended to all ops created by this `pdf`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> tf.DType:
        """The `DType` of `Tensor`s handled by this `pdf`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self) -> ztyping.ParametersType:
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self, only_floating: bool = False,
                       names: ztyping.ParamsNameOpt = None) -> List["ZfitParameter"]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
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


class ZfitParameter(ZfitObject):

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


class ZfitModel(ZfitObject):
    pass


class ZfitFunc(ZfitModel):
    @abc.abstractmethod
    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        raise NotImplementedError


class ZfitPDF(ZfitModel):

    @abc.abstractmethod
    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, name: str = "pdf") -> ztyping.XType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_extended(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def set_yield(self, value: Union[ZfitParameter, None]):
        raise NotImplementedError

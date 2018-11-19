import abc
from typing import Union

import pep487

from ..util import ztyping


class ZfitObject(pep487.ABC):

    @abc.abstractmethod
    def get_dependents(self, only_floating=False):
        raise NotImplementedError


class ZfitModel(ZfitObject):
    @abc.abstractmethod
    def get_parameters(self, only_floating=True, names=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self):
        raise NotImplementedError


class ZfitFunc(ZfitModel):
    @abc.abstractmethod
    def value(self, x: ztyping.XType, name: str = "value") -> ztyping.XType:
        raise NotImplementedError


class ZfitParameter(ZfitObject):

    @property
    @abc.abstractmethod
    def floating(self):
        raise NotImplementedError

    @floating.setter
    @abc.abstractmethod
    def floating(self):
        raise NotImplementedError

    @abc.abstractmethod
    def value(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def independent(self):
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

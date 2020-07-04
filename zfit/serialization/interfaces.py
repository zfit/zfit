#  Copyright (c) 2020 zfit

from abc import ABC, abstractmethod
from typing import Any, Dict

from zfit.serialization.zfit_repr import ZfitRepr


class ZfitSerializable(ABC):

    @abstractmethod
    def to_repr(self) -> ZfitRepr:
        ...

    @classmethod
    @abstractmethod
    def from_repr(cls, rep: ZfitRepr) -> 'ZfitSerializable':
        ...



        ...

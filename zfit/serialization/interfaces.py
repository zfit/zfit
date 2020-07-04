from abc import ABC, abstractmethod

from zfit.serialization.zfit_repr import ZfitRepr


class ZfitSerializable(ABC):

    @abstractmethod
    def to_repr(self) -> ZfitRepr:
        ...

    @classmethod
    @abstractmethod
    def from_repr(cls, rep: ZfitRepr) -> 'ZfitSerializable':
        ...


class ZfitSerializer(ABC):

    @abstractmethod
    def dumps(self, rep: ZfitRepr) -> str:
        ...

    @abstractmethod
    def loads(self, string: str) -> ZfitRepr:
        ...

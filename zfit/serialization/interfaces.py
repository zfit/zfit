#  Copyright (c) 2020 zfit


from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar

from dataclasses import dataclass, field


class ZfitSerializable(ABC):
    """Abstract interface of zfit object that is serializable."""

    @abstractmethod
    def to_repr(self) -> 'ZfitRepr':
        ...

    @classmethod
    @abstractmethod
    def from_repr(cls, rep: 'ZfitRepr') -> 'ZfitSerializable':
        ...


@dataclass
class ZfitRepr(ABC):
    """Intermediate representation of a zfit object needed to dump and load said object."""
    uid: str  # human readable, unique identifier; "type"
    fields: Dict[str, 'ZfitRepr'] = field(default_factory=dict)
    value: Any = field(default=None)

    field_structure: ClassVar[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        for field_name, field_value in self.fields.items():
            assert isinstance(field_value, self.field_structure[field_name])


class ZfitArranger(ABC):
    """Abstract interface of arranger.

    A ZfitArranger translates between :py:class:`zfit.serialization.zfit_repr.ZfitRepr` and a :py:class:`dict` that
    has been arranged to fit the structure in the serialized format."""

    @abstractmethod
    def dump(self, rep: ZfitRepr) -> Dict['str', Any]:
        return {'type': rep.uid}

    @abstractmethod
    def load(self, struct: Dict[str, Any]) -> ZfitRepr:
        return {}

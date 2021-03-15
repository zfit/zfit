#  Copyright (c) 2020 zfit


from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar, TypeVar, Type

from dataclasses import dataclass, field

T = TypeVar('T', bound='ZfitSerializable')
"""TypeVar for return type of ZfitSerializable."""


class ZfitSerializable(ABC):
    """Abstract interface of serializable zfit object.

    Register a :py:class:`ZfitRepr` for this serializable with the
    :py:function:`serialization.register` decorator."""

    @abstractmethod
    def to_repr(self) -> 'ZfitRepr':
        """Returns a :py:class:`ZfitRepr` representing this object."""
        ...

    @classmethod
    @abstractmethod
    def from_repr(cls: Type[T], repr_: 'ZfitRepr') -> T:
        """Returns a ZfitSerializable from `repr_`."""
        ...


@dataclass
class ZfitRepr(ABC):
    """Intermediate representation of a zfit object needed to dump and load said object."""

    field_structure: ClassVar[Dict[str, Any]] = field(init=False)
    """Defines the structure of the fields dictionary."""

    uid: str
    """Human readable, unique identifier"""

    fields: Dict[str, 'ZfitRepr'] = field(default_factory=dict)
    """Fields describing the objected represented by this ZfitRepr."""

    value: Any = field(default=None)
    """I don't remember what this is supposed to be."""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Simple validation method checking if self.fields complies with self.field_structure."""
        for field_name, field_value in self.fields.items():
            assert isinstance(field_value, self.field_structure[field_name])


class ZfitArranger(ABC):
    """Abstract interface of arranger.

    A ZfitArranger translates between :py:class:`zfit.serialization.zfit_repr.ZfitRepr` and a :py:class:`dict` that
    has been arranged to fit the structure in the serialized format."""

    @abstractmethod
    def dump(self, rep: 'ZfitRepr') -> Dict['str', Any]:
        ...

    @abstractmethod
    def load(self, struct: Dict[str, Any]) -> 'ZfitRepr':
        ...

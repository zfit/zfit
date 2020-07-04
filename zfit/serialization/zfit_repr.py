from abc import ABC
from typing import Dict, Any, ClassVar
from dataclasses import dataclass, field


@dataclass
class ZfitRepr(ABC):
    """Intermediate representation of a zfit object needed to dump and load said object."""
    fields: Dict[str, 'ZfitRepr'] = field(default_factory=dict)

    name: ClassVar[str] = field(init=False)
    field_structure: ClassVar[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        for field_name, field_value in self.fields.items():
            assert isinstance(field_value, self.field_structure[field_name])


def zfit_repr(name: str, field_structure: Dict[str, Any]) -> ZfitRepr:
    return type(
        name,
        (ZfitRepr,),
        dict(name=name, field_structure=field_structure)
    )

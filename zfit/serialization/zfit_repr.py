#  Copyright (c) 2020 zfit

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Union


@dataclass
class ZfitRepr:
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

    @abstractmethod
    def dump(self, rep: ZfitRepr) -> Dict['str', Any]:
        ...

    @abstractmethod
    def load(self, string: str) -> ZfitRepr:
        ...


class StrRepr(ZfitRepr):
    pass


class StrArranger(ZfitArranger):
    def dump(self, rep):
        return rep.value
        # return obj.fields['value']

    def load(self, string: str) -> StrRepr:
        return StrRepr(uid='str', value=string)


class ParamRepr(ZfitRepr):
    field_structure = {'name': StrRepr}

    # def __init_subclass__(cls, **kwargs):
    #     cls.field_structure = cls.field_structure.copy()


# def recursive_dump(structure: Dict['str': ZfitRepr]):


class ParamArranger(ZfitArranger):

    def dump(self, rep: ParamRepr) -> Dict['str', Any]:
        return {'name': StrArranger().dump(rep.fields['name'])}


class FloatRepr(ZfitRepr):
    pass


class FloatArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> str:
        return str(rep.value)

    def load(self, string: str) -> FloatRepr:
        return FloatRepr(uid='float', value=float(string))


class BoolRepr(ZfitRepr):
    pass


class BoolArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> Dict['str', Any]:
        return str(rep.value)

    def load(self, string: str) -> ZfitRepr:
        return BoolRepr(uid='bool', value=bool(string))


class IndependentParamRepr(ParamRepr):
    field_structure = {**ParamRepr.field_structure, **{
        'step_size': FloatRepr,
        'value': FloatRepr,
        # 'lower': Optional[FloatRepr],
        # 'upper': Optional[FloatRepr],
        'floating': BoolRepr

    }}


class IndependentParamArranger(ParamArranger):

    def dump(self, rep: IndependentParamRepr) -> Union[str, Dict[str, str]]:
        structure = super().dump(rep)
        structure['value'] = FloatArranger().dump(rep.fields['value'])
        step_size = rep.fields.get('step_size')
        if step_size is not None:
            structure['step_size'] = FloatArranger().dump(step_size)

        lower = rep.fields.get('lower')
        if lower is not None:
            structure['lower'] = FloatArranger().dump(lower)

        upper = rep.fields.get('upper')
        if upper is not None:
            structure['upper'] = FloatArranger().dump(upper)

        floating = rep.fields.get('floating')
        if floating is not None:
            structure['floating'] = BoolArranger().dump()(floating)
        return structure

    def load(self, string: str) -> IndependentParamRepr:
        structure = super().load(string)
        raise RuntimeError("TODO")
        return IndependentParamRepr(uid='IndepParam', fields=structure)


if __name__ == '__main__':
    import zfit

    param = zfit.Parameter('asdf', 10)
    param_repr = IndependentParamRepr(uid="IndepParam",
                                      fields={'name': StrRepr(uid='str', value=param.name),
                                              'value': FloatRepr(uid='float', value=float(param.value()))})

    dump = IndependentParamArranger().dump(param_repr)
    print(dump)

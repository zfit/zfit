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
        return {'type': rep.uid}

    @abstractmethod
    def load(self, struct: Dict[str, Any]) -> ZfitRepr:
        return {}


class StrRepr(ZfitRepr):
    pass


class StrArranger(ZfitArranger):
    def dump(self, rep):
        return rep.value
        # return obj.fields['value']

    def load(self, struct: str) -> StrRepr:
        return StrRepr(uid='str', value=struct)


class ParamRepr(ZfitRepr):
    field_structure = {'name': StrRepr}

    # def __init_subclass__(cls, **kwargs):
    #     cls.field_structure = cls.field_structure.copy()


# def recursive_dump(structure: Dict['str': ZfitRepr]):


class ParamArranger(ZfitArranger):

    def dump(self, rep: ParamRepr) -> Dict['str', Any]:
        structure = super().dump(rep)
        structure['name'] = StrArranger().dump(rep.fields['name'])
        return structure

    def load(self, struct: Dict[str, Any]) -> ZfitRepr:
        struct = struct.copy()
        rep = ParamRepr(uid=struct['type'], fields={'name': StrArranger().load(struct.pop('name'))})
        return rep


class FloatRepr(ZfitRepr):
    pass


class FloatArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> str:
        return str(rep.value)

    def load(self, struct: str) -> FloatRepr:
        return FloatRepr(uid='float', value=float(struct))


class BoolRepr(ZfitRepr):
    pass


class BoolArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> Dict['str', Any]:
        return str(rep.value)

    def load(self, struct: str) -> ZfitRepr:
        return BoolRepr(uid='bool', value=bool(struct))


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

    def load(self, struct: str) -> IndependentParamRepr:
        struct = struct.copy()

        fields = {}

        fields['value'] = FloatArranger().load(struct.pop('value'))
        step_size = struct.get('step_size')
        if step_size is not None:
            fields['step_size'] = FloatArranger().load(step_size)

        lower = struct.pop('lower', None)
        if lower is not None:
            fields['lower'] = FloatArranger().load(lower)

        upper = struct.pop('upper', None)
        if upper is not None:
            fields['upper'] = FloatArranger().load(upper)

        floating = struct.pop('floating', None)
        if floating is not None:
            fields['floating'] = BoolArranger().load()(floating)

        rep = super().load(struct)
        fields.update(rep.fields)
        return IndependentParamRepr(uid=struct['type'], fields=fields)


if __name__ == '__main__':
    import zfit

    param = zfit.Parameter('asdf', 10)
    param_repr = IndependentParamRepr(uid="IndepParam",
                                      fields={'name': StrRepr(uid='str', value=param.name),
                                              'value': FloatRepr(uid='float', value=float(param.value())),
                                              'step_size': FloatRepr(uid='float', value=float(param.step_size))
                                              })

    dump = IndependentParamArranger().dump(param_repr)
    print(dump)
    load_dump = IndependentParamArranger().load(dump)
    print(load_dump)
    # load_dump.fields['name'].value += '_loaded'
    param_loaded = zfit.Parameter(**{k: rep.value for k, rep in load_dump.fields.items()})
    print(param_loaded)

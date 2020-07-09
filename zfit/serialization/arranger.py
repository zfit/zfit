#  Copyright (c) 2020 zfit


from typing import Dict, Any, Union

from zfit.serialization.interfaces import ZfitArranger, ZfitRepr
from zfit.serialization.repr import StrRepr, ParamRepr, FloatRepr, BoolRepr, IndependentParamRepr


class StrArranger(ZfitArranger):
    def dump(self, rep):
        return rep.value
        # return obj.fields['value']

    def load(self, struct: str) -> StrRepr:
        return StrRepr(uid='str', value=struct)


class ParamArranger(ZfitArranger):

    def dump(self, rep: ParamRepr) -> Dict['str', Any]:
        structure = super().dump(rep)
        structure['name'] = StrArranger().dump(rep.fields['name'])
        return structure

    def load(self, struct: Dict[str, Any]) -> ZfitRepr:
        struct = struct.copy()
        rep = ParamRepr(uid=struct['type'], fields={'name': StrArranger().load(struct.pop('name'))})
        return rep


class FloatArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> str:
        return str(rep.value)

    def load(self, struct: str) -> FloatRepr:
        return FloatRepr(uid='float', value=float(struct))


class BoolArranger(ZfitArranger):

    def dump(self, rep: ZfitRepr) -> Dict['str', Any]:
        return str(rep.value)

    def load(self, struct: str) -> ZfitRepr:
        return BoolRepr(uid='bool', value=bool(struct))


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


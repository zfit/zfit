#  Copyright (c) 2020 zfit

from zfit.serialization.interfaces import ZfitRepr


class StrRepr(ZfitRepr):
    pass


class FloatRepr(ZfitRepr):
    pass


class BoolRepr(ZfitRepr):
    pass


class ParamRepr(ZfitRepr):
    field_structure = {'name': StrRepr}


class IndependentParamRepr(ParamRepr):
    field_structure = {**ParamRepr.field_structure, **{
        'step_size': FloatRepr,
        'value': FloatRepr,
        'lower': FloatRepr,
        'upper': FloatRepr,
        'floating': BoolRepr

    }}

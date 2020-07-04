#  Copyright (c) 2020 zfit

from zfit.serialization.interfaces import ZfitArranger
from zfit.serialization.zfit_repr import ZfitRepr


class BaseParameterSerializer(ZfitArranger):
    def dump(self, rep: ZfitRepr) -> str:
        return str(rep)

    def load(self, string: str) -> ZfitRepr:
        ...

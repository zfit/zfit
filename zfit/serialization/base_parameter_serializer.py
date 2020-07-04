from zfit.serialization.interfaces import ZfitSerializer
from zfit.serialization.zfit_repr import ZfitRepr


class BaseParameterSerializer(ZfitSerializer):
    def dumps(self, rep: ZfitRepr) -> str:
        return str(rep)

    def loads(self, string: str) -> ZfitRepr:
        ...


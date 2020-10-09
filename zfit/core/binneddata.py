#  Copyright (c) 2020 zfit
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData
from ..util.ztyping import NumericalTypeReturn


class BinnedData(BaseDimensional, ZfitBinnedData, BaseObject):  # TODO: add dtype

    def __init__(self, obs, counts, w2error, name: str = "BinnedData"):
        super().__init__(name=name, obs=obs)
        self._count = self._input_check_counts(counts)
        self._w2error = w2error

    def _input_check_counts(self, counts):  # TODO
        return counts

    def get_counts(self, bins) -> NumericalTypeReturn:
        pass

    def weights_error_squared(self) -> NumericalTypeReturn:
        pass

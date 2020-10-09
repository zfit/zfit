#  Copyright (c) 2020 zfit
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData
from .. import z
from ..util.ztyping import NumericalTypeReturn


class BinnedData(BaseDimensional, ZfitBinnedData, BaseObject):  # TODO: add dtype

    def __init__(self, obs, counts, w2error, name: str = "BinnedData"):
        super().__init__(name=name, obs=obs)
        self._counts = self._input_check_counts(counts)
        self._w2error = w2error

    @classmethod
    def from_numpy(cls, obs, counts, w2error, name):
        counts = z.convert_to_tensor(counts)
        return cls(obs=obs, counts=counts, w2error=w2error, name=name)

    def _input_check_counts(self, counts):  # TODO
        return counts

    def get_counts(self, bins) -> NumericalTypeReturn:
        return self._counts

    def weights_error_squared(self) -> NumericalTypeReturn:
        return self._w2error

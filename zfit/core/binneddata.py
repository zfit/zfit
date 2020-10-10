#  Copyright (c) 2020 zfit
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData
from .. import z
from ..util.ztyping import NumericalTypeReturn


class BinnedData(BaseDimensional, ZfitBinnedData, BaseObject):  # TODO: add dtype

    def __init__(self, obs, counts, w2error, name: str = "BinnedData"):
        if name is None:
            name = "BinnedData"
        super().__init__(name=name)
        self._space = obs
        self._counts = self._input_check_counts(counts)
        self._w2error = w2error

    @property
    def space(self):
        return self._space

    @classmethod
    def from_numpy(cls, obs, counts, w2error, name=None):
        counts = z.convert_to_tensor(counts)
        return cls(obs=obs, counts=counts, w2error=w2error, name=name)

    def _input_check_counts(self, counts):  # TODO
        return counts

    def get_counts(self, bins=None) -> NumericalTypeReturn:
        if bins is not None:
            raise WorkInProgressError
        return self._counts

    def weights_error_squared(self) -> NumericalTypeReturn:
        return self._w2error

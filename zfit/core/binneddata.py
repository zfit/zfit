#  Copyright (c) 2021 zfit
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData
from .tensorlike import register_tensor_conversion, OverloadableMixin
from .. import z
from ..util.exception import WorkInProgressError
from ..util.ztyping import NumericalTypeReturn


class BinnedData(BaseDimensional, ZfitBinnedData, BaseObject, OverloadableMixin):  # TODO: add dtype

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

    def get_counts(self, bins=None, obs=None) -> NumericalTypeReturn:
        if bins is not None:
            raise WorkInProgressError
        if obs is not None and not obs == self.obs:
            raise WorkInProgressError("Currently, reordering of axes not supported")
        return self._counts

    def weights_error_squared(self) -> NumericalTypeReturn:
        return self._w2error

    @property
    def data_range(self):
        return self.space

    def value(self):
        return self.get_counts()

    @property
    def nevents(self):
        return 42  # TODO: add sensible number

    @property
    def _approx_nevents(self):
        return 42  # TODO: add sensible number


register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)

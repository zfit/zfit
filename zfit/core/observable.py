import zfit
from zfit import convert_to_range
from zfit.core.baseobject import BaseObject
from zfit.core.interfaces import ZfitObservable


class Observable(ZfitObservable, BaseObject):

    def __init__(self, name, obs_range):
        super().__init__(name=name)
        self.data_range = None
        self.norm_range = None
        self.obs_range = obs_range

    @property
    def obs_range(self) -> "zfit.Range":
        return self._obs_range

    @obs_range.setter
    def obs_range(self, value):
        self._obs_range = convert_to_range(value)

    @property
    def norm_range(self) -> "zfit.Range":
        if self._norm_range is None:
            return self.obs_range
        else:
            return self._norm_range

    @norm_range.setter
    def norm_range(self, value):
        self._norm_range = convert_to_range(value)

    @property
    def data_range(self) -> "zfit.Range":
        if self._data_range is None:
            return self.obs_range
        else:
            return self._data_range

    @data_range.setter
    def data_range(self, value):
        self._data_range = convert_to_range(value)

    def _repr(self):
        pass  # TODO(Mayou36):


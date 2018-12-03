from typing import Iterable, Tuple, List

import zfit
from zfit.core.limits import convert_to_range
from zfit.core.baseobject import BaseObject
from zfit.core.interfaces import ZfitObservable
from zfit.util.container import convert_to_container



class Observable(ZfitObservable, BaseObject):

    def __init__(self, names, lower, upper):
        names = convert_to_container(names, container=tuple)
        # TODO(Mayou36): good name?
        name = "_".join(names)
        super().__init__(name=name)
        self._names = names
        self.data_range = None
        self.norm_range = None
        self.obs_range = ranges

    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    def get_subspace(self, names: str = None) -> "ZfitObservable":
        pass

    @property
    def obs_range(self) -> "zfit.Range":
        return self._obs_range

    @obs_range.setter
    def obs_range(self, value):
        self._obs_range = convert_to_range(value)

    # def _check_input_ranges(self, ranges: List["zfit.Range"]) -> Tuple["zfit.Range"]:
    #
    #     if isinstance(ranges, list):
    #         ranges = [convert_to_range()]
    #
    #     ranges = convert_to_container(ranges, container=tuple)
    #     self.obs_range = sum(ranges)
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

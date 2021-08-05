#  Copyright (c) 2021 zfit

from __future__ import annotations

import boost_histogram as bh
import hist
import tensorflow_probability as tfp

from zfit.z import numpy as znp
from .baseobject import BaseObject
from .dimension import BaseDimensional
from .interfaces import ZfitBinnedData
from .tensorlike import register_tensor_conversion, OverloadableMixin, OverloadableMixinValues
from .. import z, Space
from ..util.exception import WorkInProgressError
from ..util.ztyping import NumericalTypeReturn


class BinnedDataV1(BaseDimensional, ZfitBinnedData, BaseObject, OverloadableMixin):  # TODO: add dtype

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


register_tensor_conversion(BinnedDataV1, name='BinnedData', overload_operators=True)


@tfp.experimental.auto_composite_tensor()
class BinnedHolder(tfp.experimental.AutoCompositeTensor):
    def __init__(self, space, values, variances):
        self.space = space
        self.values = values
        self.variances = variances


@tfp.experimental.auto_composite_tensor()
class BinnedData(tfp.experimental.AutoCompositeTensor, OverloadableMixinValues):

    def __init__(self, holder):
        self.holder: BinnedHolder = holder

    @classmethod
    def from_tensor(cls, space, values, variances=None):
        values = znp.asarray(values)
        if variances is not None:
            variances = znp.asarray(variances)
        return cls(BinnedHolder(space=space, values=values, variances=variances))

    @classmethod
    def from_hist(cls, hist: hist.NamedHist) -> BinnedData:
        space = Space(binning=hist.axes)
        values = znp.asarray(hist.values(flow=True))
        variances = znp.asarray(hist.variances(flow=True))
        holder = BinnedHolder(space=space, values=values, variances=variances)
        return cls(holder)

    def to_hist(self) -> hist.NamedHist:
        h = hist.NamedHist(*self.holder.space.binning, storage=bh.storage.Weight)
        h.view(flow=True).value = self.values()
        h.view(flow=True).variance = self.variances()
        return h

    def _to_boost_histogram_(self):
        h = bh.Histogram(*self.holder.space.binning, storage=bh.storage.Weight)
        h.view(flow=True).value = self.values()
        h.view(flow=True).variance = self.variances()

        # h[...] = np.stack([self.values(), self.variances()], axis=-1)
        return h

    @property
    def space(self):
        return self.holder.space

    @property
    def axes(self):
        return self.space.binning

    def values(self):
        return self.holder.values

    def variances(self):
        return self.holder.variances

    def counts(self):
        return self.values()


register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)

#  Copyright (c) 2021 zfit

from __future__ import annotations

import boost_histogram as bh
import hist
import tensorflow_probability as tfp
import zfit.core.tensorlike as tensorlike

from zfit.z import numpy as znp
from zfit.core.baseobject import BaseObject
from zfit.core.dimension import BaseDimensional
from zfit.core.interfaces import ZfitBinnedData
from zfit.core.tensorlike import register_tensor_conversion, OverloadableMixin, OverloadableMixinValues
from zfit import z, Space
from zfit.util.exception import WorkInProgressError
from zfit.util.ztyping import NumericalTypeReturn


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


# class ZfitMixedData:
#     def values(self):


flow = False  # TODO: track the flow or not?


@tfp.experimental.auto_composite_tensor()
class BinnedData(tfp.experimental.AutoCompositeTensor, OverloadableMixinValues, ZfitBinnedData):

    def __init__(self, holder):
        self.holder: BinnedHolder = holder

    @classmethod
    def from_tensor(cls, space, values, variances=None):  # TODO: add overflow bins if needed
        values = znp.asarray(values)
        if variances is not None:
            variances = znp.asarray(variances)
        return cls(BinnedHolder(space=space, values=values, variances=variances))

    @classmethod
    def from_hist(cls, hist: hist.NamedHist) -> BinnedData:
        space = Space(binning=hist.axes)
        values = znp.asarray(hist.values(flow=flow))
        variances = znp.asarray(hist.variances(flow=flow))
        holder = BinnedHolder(space=space, values=values, variances=variances)
        return cls(holder)

    @property
    def n_obs(self) -> int:
        return self.space.n_obs

    @property
    def obs(self):
        return self.space.obs

    def to_hist(self) -> hist.NamedHist:
        h = hist.NamedHist(*self.holder.space.binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values(flow=flow)
        h.view(flow=flow).variance = self.variances(flow=flow)
        return h

    def _to_boost_histogram_(self):
        h = bh.Histogram(*self.holder.space.binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values(flow=flow)
        h.view(flow=flow).variance = self.variances(flow=flow)

        # h[...] = np.stack([self.values(), self.variances()], axis=-1)
        return h

    @property
    def space(self):
        return self.holder.space

    @property
    def axes(self):
        return self.space.binning

    def values(self, flow=False):
        vals = self.holder.values
        # if not flow:
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)
        return vals

    def variances(self, flow=False):
        vals = self.holder.variances
        # if not flow:
        #     shape = tf.shape(vals)
        #     vals = tf.slice(vals, znp.ones_like(shape), shape - 2)
        return vals

    def counts(self):
        return self.values()

    # dummy
    @property
    def data_range(self):
        return self.space

    @property
    def nevents(self):
        return znp.sum(self.values())

    @property
    def _approx_nevents(self):
        return znp.sum(self.values())


tensorlike.register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)

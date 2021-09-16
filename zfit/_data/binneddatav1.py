#  Copyright (c) 2021 zfit

from __future__ import annotations

import boost_histogram as bh
import hist
import tensorflow as tf

from zfit._variables.axis import histaxes_to_binning, binning_to_histaxes
from zfit.core.interfaces import ZfitBinnedData
from ..util.exception import ShapeIncompatibleError
from zfit.z import numpy as znp


# @tfp.experimental.auto_composite_tensor()
class BinnedHolder(
    # tfp.experimental.AutoCompositeTensor
):
    def __init__(self, space, values, variances):
        self._check_init_values(space, values, variances)
        self.space = space
        self.values = values
        self.variances = variances

    def _check_init_values(self, space, values, variances):
        value_shape = tf.shape(values)
        edges_shape = znp.array([tf.shape(znp.reshape(edge, (-1,)))[0] for edge in space.binning.edges])
        values_rank = value_shape.shape[0]
        if variances is not None:
            variances_shape = tf.shape(variances)
            variances_rank = variances_shape.shape[0]
            if values_rank != variances_rank:
                raise ShapeIncompatibleError(
                    f"Values {values} and variances {variances} differ in rank: {values_rank} vs {variances_rank}")
            tf.assert_equal(variances_shape, value_shape,
                            message=f"Variances and values do not have the same shape:"
                                    f" {variances_shape} vs {value_shape}")
        binning_rank = len(space.binning.edges)
        if binning_rank != values_rank:
            raise ShapeIncompatibleError(f"Values and binning  differ in rank: {values_rank} vs {binning_rank}")
        tf.assert_equal(edges_shape - 1, value_shape,
                        message=f"Edges (minus one) and values do not have the same shape:"
                                f" {edges_shape} vs {value_shape}")

    def with_obs(self, obs):
        space = self.space.with_obs(obs)
        values = move_axis_obs(self.space, space, self.values)
        variances = self.variances
        if variances is not None:
            variances = move_axis_obs(self.space, space, self.variances)
        return type(self)(space=space, values=values, variances=variances)


def move_axis_obs(original, target, values):
    new_axes = [original.obs.index(ob) for ob in target.obs]
    values = znp.moveaxis(values, tuple(range(target.n_obs)), new_axes)
    return values


flow = False  # TODO: track the flow or not?


# @tfp.experimental.auto_composite_tensor()
class BinnedDataV1(ZfitBinnedData,
                   # tfp.experimental.AutoCompositeTensor, OverloadableMixinValues, ZfitBinnedData
                   ):

    def __init__(self, holder):
        self.holder: BinnedHolder = holder

    @classmethod
    def from_tensor(cls, space, values, variances=None):  # TODO: add overflow bins if needed
        values = znp.asarray(values)
        if variances is not None:
            variances = znp.asarray(variances)
        return cls(BinnedHolder(space=space, values=values, variances=variances))

    @classmethod
    def from_hist(cls, hist: hist.NamedHist) -> BinnedDataV1:
        from zfit import Space
        space = Space(binning=histaxes_to_binning(hist.axes))
        values = znp.asarray(hist.values(flow=flow))
        variances = hist.variances(flow=flow)
        if variances is not None:
            variances = znp.asarray(variances)
        holder = BinnedHolder(space=space, values=values, variances=variances)
        return cls(holder)

    def with_obs(self, obs) -> BinnedDataV1:
        return type(self)(self.holder.with_obs(obs))

    @property
    def kind(self):
        return "COUNT"

    @property
    def n_obs(self) -> int:
        return self.space.n_obs

    @property
    def obs(self):
        return self.space.obs

    def to_hist(self) -> hist.NamedHist:
        binning = binning_to_histaxes(self.holder.space.binning)
        h = hist.NamedHist(*binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values(flow=flow)
        h.view(flow=flow).variance = self.variances(flow=flow)
        return h

    def _to_boost_histogram_(self):
        binning = binning_to_histaxes(self.holder.space.binning)
        h = bh.Histogram(*binning, storage=bh.storage.Weight())
        h.view(flow=flow).value = self.values(flow=flow)
        h.view(flow=flow).variance = self.variances(flow=flow)
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

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def to_unbinned(self):
        meshed_center = znp.meshgrid(*self.axes.centers, indexing='ij')
        flat_centers = [znp.reshape(center, (-1,)) for center in meshed_center]
        centers = znp.stack(flat_centers, axis=-1)
        flat_weights = znp.reshape(self.values(flow=False), (-1,))
        space = self.space.copy(binning=None)
        from zfit import Data
        return Data.from_tensor(obs=space, tensor=centers, weights=flat_weights)

# tensorlike.register_tensor_conversion(BinnedData, name='BinnedData', overload_operators=True)

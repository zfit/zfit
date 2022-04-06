#  Copyright (c) 2022 zfit
import warnings

import hist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from zfit import z
from zfit.core.interfaces import ZfitData, ZfitRectBinning
from zfit.util.ztyping import XTypeInput


def rect_binning_histogramdd(data: XTypeInput, binning: ZfitRectBinning):
    if isinstance(data, ZfitData):
        data = data.value()
    return histogramdd(sample=data, bins=binning.get_edges())


def histogramdd(sample, bins=10, range=None, weights=None, density=None):
    out_dtype = [tf.float64, tf.float64]
    if isinstance(sample, ZfitData):
        sample = sample.value()
        n_obs = sample.n_obs
    else:
        sample = z.convert_to_tensor(sample)
        n_obs = sample.shape[-1]

    none_tensor = tf.constant("NONE_TENSOR", shape=(), name="none_tensor")
    inputs = [sample, bins, range, weights]
    inputs_cleaned = [inp if inp is not None else none_tensor for inp in inputs]

    def histdd(sample, bins, range, weights):
        kwargs = {"sample": sample, "bins": bins, "range": range, "weights": weights}
        new_kwargs = {}
        for key, value in kwargs.items():
            value = value
            if value == b"NONE_TENSOR":
                value = None

            new_kwargs[key] = value
        return np.histogramdd(**new_kwargs, density=density)

    bincounts, *edges = tf.numpy_function(
        func=histdd, inp=inputs_cleaned, Tout=out_dtype
    )
    bincounts.set_shape(shape=(None,) * n_obs)
    # edges = [edge.set_shape(shape=(None)) for edge in edges]
    return bincounts, edges


def unbinned_to_hist_eager(values, edges, weights=None):
    if weights is not None and weights.shape == () and None in weights:
        weights = None
    binning = [
        hist.axis.Variable(np.reshape(edge, (-1,)), flow=False) for edge in edges
    ]
    h = hist.Hist(*binning, storage=hist.storage.Weight())
    h.fill(*(values[:, i] for i in range(values.shape[1])), weight=weights)

    return znp.array(h.values(flow=False), znp.float64), znp.array(
        h.variances(flow=False), znp.float64
    )


def unbinned_to_binned(data, space, binned_class=None):
    if binned_class is None:
        from zfit._data.binneddatav1 import BinnedData

        binned_class = BinnedData
    values = data.value()
    weights = data.weights
    if weights is not None:
        weights = znp.array(weights)
    edges = tuple(space.binning.edges)
    values, variances = tf.numpy_function(
        unbinned_to_hist_eager,
        inp=[values, edges, weights],
        Tout=[tf.float64, tf.float64],
    )
    binned = binned_class.from_tensor(space=space, values=values, variances=variances)
    return binned


def unbinned_to_binindex(data, space, flow=False):
    if flow:
        warnings.warn(
            "Flow currently not fully supported. Values outside the edges are all 0."
        )
    values = [znp.reshape(data.value(ob), (-1,)) for ob in space.obs]
    edges = [znp.reshape(edge, (-1,)) for edge in space.binning.edges]
    bins = [tfp.stats.find_bins(x=val, edges=edge) for val, edge in zip(values, edges)]
    stacked_bins = znp.stack(bins, axis=-1)
    if flow:
        stacked_bins += 1
        bin_is_nan = tf.math.is_nan(stacked_bins)
        zeros = znp.zeros_like(stacked_bins)
        binindices = znp.where(bin_is_nan, zeros, stacked_bins)
        stacked_bins = znp.asarray(binindices, dtype=znp.int32)
    return stacked_bins


def midpoints_from_hist(bincounts, edges):  # TODO: implement correctly, old
    """Calculate the midpoints of a hist and return the non-zero entries, non-zero bincounts and indices.

    Args:
        bincounts: Tensor with shape (nbins_0, ..., nbins_n) with n being the dimension.
        edges: Tensor with shape (n_obs, nbins + 1) holding the position of the edges, assuming a rectangular grid.
    Returns:
        bincounts: the bincounts that are non-zero in a 1-D array corresponding to the indices and the midpoints
        midpoints: the coordinates of the midpoint of each bin with shape (nbincounts, n_obs)
        indices: original position in the bincounts from the input
    """
    bincounts = z.convert_to_tensor(bincounts)
    edges = z.convert_to_tensor(edges)

    midpoints = (edges[:, :-1] + edges[:, 1:]) / 2.0
    midpoints_grid = tf.stack(
        tf.meshgrid(*tf.unstack(midpoints), indexing="ij"), axis=-1
    )
    bincounts_nonzero_index = tf.where(bincounts)
    bincounts_nonzero = tf.gather_nd(bincounts, indices=bincounts_nonzero_index)
    midpoints_nonzero = tf.gather_nd(midpoints_grid, indices=bincounts_nonzero_index)
    return bincounts_nonzero, midpoints_nonzero, bincounts_nonzero_index

#  Copyright (c) 2024 zfit
from __future__ import annotations

import warnings

import hist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitSpace


def unbinned_to_hist_eager_edgesweightsargs(values, *edges_weights):
    """Same as `unbinned_to_hist_eager` but with the edges and weights as positional arguments.

    This is needed to circumvent the limitation of `tf.numpy_function` that only allows
    numpy arrays as positional arguments and not structures of numpy arrays, such as edges are.

    Args:
        values:
        *edges_weights:

    Returns:
    """
    *edges, weights = edges_weights
    return unbinned_to_hist_eager(values, edges, weights=weights)


def unbinned_to_hist_eager(values, edges, weights=None):
    """Convert an unbinned dataset to a binned dataset in eager mode.

    Args:
        values: Unbinned dataset to convert.
        edges: Edges of the bins.
        weights: Event weights.

    Returns:
        binned_data: Binned dataset.
    """
    if weights is not None and weights.shape == () and None in weights:
        weights = None
    binning = [hist.axis.Variable(np.reshape(edge, (-1,)), flow=False) for edge in edges]
    h = hist.Hist(*binning, storage=hist.storage.Weight())
    h.fill(*(values[:, i] for i in range(values.shape[1])), weight=weights)

    return znp.array(h.values(flow=False), znp.float64), znp.array(h.variances(flow=False), znp.float64)


def unbinned_to_binned(data, space, binned_class=None, initkwargs=None):
    """Convert an unbinned dataset to a binned dataset.

    Args:
        data: Unbinned dataset to convert.
        space: Space to bin the data in.
        binned_class: Class to use for the binned dataset. Defaults to `BinnedData`.

    Returns:
        binned_data: Binned dataset of type `binned_class`.
    """
    if binned_class is None:
        from zfit._data.binneddatav1 import BinnedData

        binned_class = BinnedData
    if not isinstance(space, ZfitSpace):
        try:
            space = data.space.with_binning(space)
        except Exception as error:
            msg = (
                f"The space provided is not a valid space for the data. "
                f"Either provide a valid space or a binning. "
                f"The error was: {error}"
            )
            raise ValueError(msg) from error

    values = data.value()
    weights = data.weights
    if weights is not None:
        weights = znp.array(weights)
    edges = tuple(space.binning.edges)
    values, variances = tf.numpy_function(
        unbinned_to_hist_eager_edgesweightsargs,
        inp=[values, *edges, weights],
        Tout=[tf.float64, tf.float64],
    )
    return binned_class.from_tensor(space=space, values=values, variances=variances, **(initkwargs or {}))


def unbinned_to_binindex(data, space, flow=False):
    """Calculate the bin index of each data point.

    Args:
        data: Data to calculate the bin index for.
        space: Defines the binning.
        flow: Whether to include the underflow and overflow bins.

    Returns:
        binindex: Tensor with shape (ndata, n_obs) holding the bin index of each data point.
    """
    if flow:
        warnings.warn(
            "Flow currently not fully supported. Values outside the edges are all 0.", UserWarning, stacklevel=2
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

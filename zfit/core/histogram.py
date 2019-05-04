#  Copyright (c) 2019 zfit
import tensorflow as tf
import numpy as np

from .interfaces import ZfitData


def histogramdd(sample, bins=10, range=None, weights=None,
                density=None):
    out_dtype = [tf.float64, tf.float64]
    if isinstance(sample, ZfitData):
        sample = sample.value()
        n_obs = sample.n_obs
    else:
        n_obs = sample.shape[-1].value

    none_tensor = tf.constant("NONE_TENSOR", shape=(), name="none_tensor")
    inputs = [sample, bins, range, weights]
    inputs_cleaned = [inp if inp is not None else none_tensor for inp in inputs]

    def histdd(sample, bins, range, weights):
        kwargs = {"sample": sample,
                  "bins": bins,
                  "range": range,
                  "weights": weights}
        new_kwargs = {}
        for key, value in kwargs.items():
            value = value.numpy()
            if value == b"NONE_TENSOR":
                value = None

            new_kwargs[key] = value
        return np.histogramdd(**new_kwargs, density=density)

    bincounts, edges = tf.py_function(func=histdd, inp=inputs_cleaned, Tout=out_dtype)
    bincounts.set_shape(shape=(None,) * n_obs)
    edges.set_shape(shape=(n_obs, None))
    return bincounts, edges


def midpoints_from_hist(bincounts, edges):
    """Calculate the midpoints of a hist and return the non-zero entries, non-zero bincounts and indices.

    Args:
        bincounts: Tensor with shape (nbins_0, ..., nbins_n) with n being the dimension.
        edges: Tensor with shape (n_obs, nbins + 1) holding the position of the edges, assuming a rectangular grid.

    Returns:
        bincounts: the bincounts that are non-zero in a 1-D array corresponding to the indices and the midpoints
        midpoints: the coordinates of the midpoint of each bin with shape (nbincounts, n_obs)
        indices: original position in the bincounts from the input
    """

    midpoints = (edges[:, :-1] + edges[:, 1:]) / 2.
    midpoints_grid = tf.stack(tf.meshgrid(*tf.unstack(midpoints), indexing='ij'), axis=-1)
    bincounts_nonzero_index = tf.where(bincounts)
    bincounts_nonzero = tf.gather_nd(bincounts, indices=bincounts_nonzero_index)
    midpoints_nonzero = tf.gather_nd(midpoints_grid, indices=bincounts_nonzero_index)
    return bincounts_nonzero, midpoints_nonzero, bincounts_nonzero_index

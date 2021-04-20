#  Copyright (c) 2021 zfit
from typing import List

import boost_histogram as bh
import numpy as np
import tensorflow as tf

from zfit import z
from zfit.core.interfaces import ZfitData, ZfitRectBinning
from zfit.util.ztyping import XTypeInput


class RectBinning(ZfitRectBinning):

    def __init__(self, binnings):
        super().__init__()
        self._binnings = binnings

    def get_binnings(self) -> List[bh.axis.Axis]:
        return self._binnings

    def get_edges(self) -> np.array:
        # edges = np.meshgrid(*[binning.edges for binning in self.get_binnings()], indexing='ij')
        edges = [binning.edges for binning in self.get_binnings()]
        return edges


def rect_binning_histogramdd(data: XTypeInput, binning: ZfitRectBinning):
    if isinstance(data, ZfitData):
        data = data.value()
    return histogramdd(sample=data, bins=binning.get_edges())


def histogramdd(sample, bins=10, range=None, weights=None,
                density=None):
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
        kwargs = {"sample" : sample,
                  "bins"   : bins,
                  "range"  : range,
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

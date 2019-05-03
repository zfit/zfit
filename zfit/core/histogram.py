#  Copyright (c) 2019 zfit
import tensorflow as tf
import numpy as np

from .interfaces import ZfitData


def histogramdd(sample, bins=10, range=None, weights=None,
                density=None):
    out_dtype = [tf.float64, tf.float64]
    if isinstance(sample, ZfitData):
        sample = sample.value()

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
            print(f"key {key}: {value}")
        return np.histogramdd(**new_kwargs, density=density)

    return tf.py_function(func=histdd, inp=inputs_cleaned, Tout=out_dtype)


def midpoints_from_hist(bincounts, edges):
    midpoints = (edges[:-1] + edges[1:]) / 2
    return midpoints

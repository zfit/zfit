#  Copyright (c) 2022 zfit
import tensorflow as tf

import zfit.z.numpy as znp


def test_z_numpy_ndarray_is_tensorflow_tensor():
    """In tensorflow 2.4.1 tf.experimental.numpy.ndarray was a wrapper around tf.Tensor.

    Now this concept seems to have been scratched and tf.experimental.numpy.ndarray is just an alias for tf.Tensor. See
    the commit history of
    https://github.com/tensorflow/tensorflow/commits/master/tensorflow/python/ops/numpy_ops/np_arrays.py
    """
    assert znp.ndarray is tf.Tensor
    assert isinstance(znp.array(1), tf.Tensor)
    assert isinstance(znp.sum(znp.array(0)), tf.Tensor)

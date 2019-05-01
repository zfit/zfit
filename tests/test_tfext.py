# deactivating CUDA capable gpus
from zfit.ztf.tools import _auto_upcast
from zfit.core.testing import setup_function, teardown_function, tester

suppress_gpu = False
if suppress_gpu:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import tensorflow as tf
import numpy as np

import zfit.core.math
from zfit import ztf

prec = 0.00001


def test_polynomial():
    coeffs = [5.3, 1.2, complex(1.3, 0.4), -42, 32.4, 529.3, -0.93]
    x = tf.placeholder(tf.complex128)
    true_dict = {'x': 5.}
    feed_dict = {x: true_dict['x']}
    polynom_tf = zfit.core.math.poly_complex(*(coeffs + [x]))  # py34 comp: *x, y does not work
    polynom_np = np.polyval(coeffs[::-1], true_dict['x'])

    result = zfit.run(polynom_tf, feed_dict=feed_dict)
    assert result == pytest.approx(polynom_np, rel=prec)


def test_auto_upcast():
    tensor_from_f32 = _auto_upcast(tf.constant(5, dtype=tf.float32))
    tensor_from_f64 = _auto_upcast(tf.constant(5, dtype=tf.float64))
    assert tensor_from_f32.dtype == tf.float64
    assert tensor_from_f64.dtype == tf.float64

    tensor_from_i32 = _auto_upcast(tf.constant(5, dtype=tf.int32))
    tensor_from_i64 = _auto_upcast(tf.constant(5, dtype=tf.int64))
    assert tensor_from_i32.dtype == tf.int64
    assert tensor_from_i64.dtype == tf.int64

    tensor_from_c64 = _auto_upcast(tf.constant(5., dtype=tf.complex64))
    tensor_from_c128 = _auto_upcast(tf.constant(5., dtype=tf.complex128))
    assert tensor_from_c64.dtype == tf.complex128
    assert tensor_from_c128.dtype == tf.complex128

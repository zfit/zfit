
# deactivating CUDA capable gpus

suppress_gpu = True
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
    """Empty test."""
    coeffs = [5.3, 1.2, complex(1.3, 0.4), -42, 32.4, 529.3, -0.93]
    x = tf.placeholder(tf.complex128)
    true_dict = {'x': 5.}
    feed_dict = {x: true_dict['x']}
    polynom_tf = zfit.core.math.poly_complex(*(coeffs + [x]))  # py34 comp: *x, y does not work
    polynom_np = np.polyval(coeffs[::-1], true_dict['x'])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(polynom_tf, feed_dict=feed_dict)
        assert result == pytest.approx(polynom_np, rel=prec)

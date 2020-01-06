#  Copyright (c) 2020 zfit
import tensorflow as tf


@tf.function(autograph=False, experimental_relax_shapes=True)
def allclose(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.less_equal(tf.abs(x - y), tf.abs(y) * rtol + atol))

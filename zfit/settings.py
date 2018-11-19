import numpy as np
import tensorflow as tf

from zfit.util.container import DotDict

# legacy settings
LEGACY_MODE = True


# Use double precision throughout


def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)


types = DotDict({'float': tf.float64,
                 'complex': tf.complex128,
                 'int': tf.int64,
                 tf.float16: tf.float64,
                 tf.float32: tf.float64,
                 tf.float64: tf.float64,
                 tf.complex64: tf.complex128,
                 tf.complex128: tf.complex128,
                 tf.int8: tf.int64,
                 tf.int16: tf.int64,
                 tf.int32: tf.int64,
                 tf.int64: tf.int64,
                 })

options = DotDict({'epsilon': 1e-8})

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

# Use double precision throughout
fptype = tf.float64

# Use double precision throughout

ctype = tf.complex128


def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)

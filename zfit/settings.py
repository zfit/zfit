from __future__ import print_function, division, absolute_import

import numpy as np

# legacy settings
import tensorflow as tf

from zfit.util.container import dotdict

LEGACY_MODE = True


# Use double precision throughout


def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)


types = dotdict({'float': tf.float64,
                 'complex': tf.complex128})

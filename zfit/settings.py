import numpy as np

# legacy settings
import tensorflow as tf

from zfit.util.container import DotDict

LEGACY_MODE = True


# Use double precision throughout


def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)


types = DotDict({'float': tf.float64,
                 'complex': tf.complex128})

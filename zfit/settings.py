import numpy as np
import tensorflow as tf

from .util.container import DotDict
from .util.execution import RunManager

run = RunManager()


def set_seed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

_verbosity = 5

def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity

def get_verbosity():
    return _verbosity


ztypes = DotDict({'float': tf.float64,
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
                 'auto_upcast': True,
                  })

options = DotDict({'epsilon': 1e-8})


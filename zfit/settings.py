#  Copyright (c) 2021 zfit

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from .util.execution import RunManager

run = RunManager()


def set_seed(seed):
    """Set random seed for numpy."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


_verbosity = 0


def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity


def get_verbosity():
    return _verbosity


ztypes = DotMap({'float': tf.float64,
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

options = DotMap({'epsilon': 1e-8,
                  'numerical_grad': None,
                  'advanced_warning': True,
                  'changed_warning': True})

advanced_warnings = DotMap({
    'sum_extended_frac': True,
    'exp_shift': True,
    'py_func_autograd': True,
    'inconsistent_fitrange': True,
    'extended_in_UnbinnedNLL': True,
    'all': True,
})

changed_warnings = DotMap({
    'new_sum': True,
    'all': True,
})

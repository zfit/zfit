#  Copyright (c) 2021 zfit

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
    tf.random.set_seed(seed)


_verbosity = 5


def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity


def get_verbosity():
    return _verbosity


def use_vectorized_parameter(vectorized=None):
    from zfit import Parameter
    from zfit.core.parameter import WrappedVariable, WrappedVectorizedVariable
    if vectorized is None:
        vectorized = True  # default
    if vectorized:
        Parameter._set_base_variable(WrappedVectorizedVariable)
    else:
        Parameter._set_base_variable(WrappedVariable)


def parameters_are_vectorized():
    from zfit import Parameter
    from zfit.core.parameter import WrappedVectorizedVariable

    return Parameter._variable_class is WrappedVectorizedVariable


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

options = DotDict({'epsilon': 1e-8,
                   'numerical_grad': None,
                   'advanced_warning': True,
                   'changed_warning': True})

advanced_warnings = DotDict({
    'sum_extended_frac': True,
    'exp_shift': True,
    'py_func_autograd': True,
    'inconsistent_fitrange': True,
    'extended_in_UnbinnedNLL': True,
    'all': True,
})

changed_warnings = DotDict({
    'new_sum': True,
    'all': True,
})

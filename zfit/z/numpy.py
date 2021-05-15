"""Numpy like interface for math functions and arrays.

This module is intended to replace tensorflow specific methods and datastructures with equivalent or similar versions
in the numpy api. This should help make zfit as a project portable to alternatives of tensorflow should it be
necessary in the future.
At the moment it is simply an alias for the numpy api of tensorflow.
See https://www.tensorflow.org/guide/tf_numpy for more a guide to numpy api in tensorflow.
See https://www.tensorflow.org/api_docs/python/tf/experimental/numpy for the complete numpy api in tensorflow.

Recommended way of importing:
>>> Import zfit.z.numpy as znp
"""

from tensorflow.experimental.numpy import *

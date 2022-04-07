"""Module is a backend which consists of two parts, a control layer and a numpy-like layer.

z itself wraps (or will in the future) control flow such as ``cond``, ``while_loop`` etc. that can be found in
different libraries supporting compiled, vectorized computation such as JAX and TensorFlow.

Numpy is a Numpy like interface for math functions and arrays.

This module is intended to replace tensorflow specific methods and datastructures with equivalent or similar versions
in the numpy api. This should help make zfit as a project portable to alternatives of tensorflow should it be
necessary in the future.
At the moment it is simply an alias for the numpy api of tensorflow.
See https://www.tensorflow.org/guide/tf_numpy for more a guide to numpy api in tensorflow.
See https://www.tensorflow.org/api_docs/python/tf/experimental/numpy for the complete numpy api in tensorflow.

Recommended way of importing:
>>> Import zfit.z.numpy as znp
"""

#  Copyright (c) 2022 zfit

# TODO: dymamic imports?
# import tensorflow.experimental.numpy as _tnp  # this way we do get the autocompletion
# numpy = _tnp  # for static code analysis
# import sys
# sys.modules['zfit.z.numpy'] = _tnp


from . import math, random, unstable
from .tools import _get_ndims
from .wrapping_tf import (
    check_numerics,
    complex,
    convert_to_tensor,
    exp,
    pow,
    random_normal,
    random_uniform,
    reduce_prod,
    reduce_sum,
    sqrt,
    square,
)
from .zextension import (
    abs_square,
    constant,
    function as function,
    function_sampling,
    function_tf_input,
    nth_pow,
    pi,
    py_function,
    run_no_nan,
    safe_where,
    stack_x,
    to_complex,
    to_real,
    unstack_x,
)

# numpy = _tnp

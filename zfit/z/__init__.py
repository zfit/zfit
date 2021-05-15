"""Thin wrapper around tensorflow.

z is a zfit TensorFlow version, that wraps TF while adding some conveniences, basically using a different default
dtype (`zfit.ztypes`). In addition, it expands TensorFlow by adding a few convenient functions helping to deal with
`NaN`s and similar.

Some function are already wrapped, others are not. Best practice is to use `z` whenever possible and `tf` for the rest.
"""

#  Copyright (c) 2021 zfit

from . import math, random, unstable
from .wrapping_tf import (check_numerics, complex, convert_to_tensor, exp, pow,
                          random_normal, random_uniform, reduce_prod,
                          reduce_sum, sqrt, square)
from .zextension import abs_square, constant
from .zextension import function as function
from .zextension import (function_sampling, function_tf_input, nth_pow, pi,
                         py_function, run_no_nan, safe_where, stack_x,
                         to_complex, to_real, unstack_x)

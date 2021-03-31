"""z is a zfit TensorFlow version, that wraps TF while adding some conveniences, basically using a different default
dtype (`zfit.ztypes`). In addition, it expands TensorFlow by adding a few convenient functions helping to deal with
`NaN`s and similar.

Some function are already wrapped, others are not. Best practice is to use `z` whenever possible and `tf` for the rest.
"""

#  Copyright (c) 2021 zfit

# fill the following in to the namespace for (future) wrapping

# doesn't work below because of autoimport... probably anytime in the Future :)
# from tensorflow import *  # Yes, this is wanted. Yields an equivalent z BUT we COULD wrap it :)
# _module_dict = tensorflow.__dict__
# try:
#     _to_import = tensorflow.__all__
# except AttributeError:
#     _to_import = [name for name in _module_dict if not name.startswith('_')]
#
# __all__ = _to_import
# del tensorflow
# _imported = {}
# _failed_imports = []
# for _name in _to_import:
#     try:
#         _imported[_name] = _module_dict[_name]
#     except KeyError as error:
#         _failed_imports.append(_name)
# if _failed_imports:
#     warnings.warn("The following modules/attributes from TensorFlow could NOT be imported:\n{}".format(
#     _failed_imports))
# globals().update(_imported)
#
# del _imported, _failed_imports, _to_import, _module_dict

# same as in TensorFlow, wrapped


# noinspection PyUnresolvedReferences
import tensorflow.experimental.numpy as numpy

from . import math, random, unstable
from .wrapping_tf import (check_numerics, complex, convert_to_tensor, exp, pow,
                          random_normal, random_uniform, reduce_prod,
                          reduce_sum, sqrt, square)
from .zextension import abs_square, constant
from .zextension import function as function
from .zextension import (function_sampling, function_tf_input, nth_pow, pi,
                         py_function, run_no_nan, safe_where, stack_x,
                         to_complex, to_real, unstack_x)

# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

#  Copyright (c) 2019 zfit

from pkg_resources import get_distribution

__version__ = get_distribution(__name__).version

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2018, zfit"
__status__ = "Beta"

__author__ = "Jonas Eschle"
__maintainer__ = "zfit"
__email__ = 'zfit@physik.uzh.ch'
__credits__ = ["Jonas Eschle <Jonas.Eschle@cern.ch>",
               "Albert Puig <apuignav@gmail.com",
               "Rafael Silva Coutinho <rafael.silva.coutinho@cern.ch>", ]

__all__ = ["ztf", "z", "constraint", "pdf", "minimize", "loss", "core", "data", "func",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]

#  Copyright (c) 2019 zfit
import tensorflow.compat.v1 as tf

# tf.enable_resource_variables()  # forward compat
# tf.enable_v2_tensorshape()  # forward compat
tf.enable_v2_behavior()
tf.disable_eager_execution()

from . import ztf  # legacy
from . import ztf as z
from .settings import ztypes

# tf.get_variable_scope().set_use_resource(True)
# tf.get_variable_scope().set_dtype(ztypes.float)

from . import constraint, pdf, minimize, loss, core, data, func, param
from .core.parameter import Parameter, ComposedParameter, ComplexParameter, convert_to_parameter
from .core.limits import Space, convert_to_space, supports
from .core.data import Data

from .settings import run

# EOF

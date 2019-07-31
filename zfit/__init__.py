# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

__version__ = '0.3.5'

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2018, zfit"
__status__ = "Beta"

__author__ = "Jonas Eschle"
__maintainer__ = "zfit"
__email__ = 'zfit@physik.uzh.ch'
__credits__ = ["Jonas Eschle <jonas.eschle@cern.ch>",
               "Albert Puig <albert.puig@cern.ch",
               "Rafael Silva Coutinho <rafael.silva.coutinho@cern.ch>", ]

__all__ = ["ztf", "z", "constraint", "pdf", "minimize", "loss", "core", "data", "func",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]

#  Copyright (c) 2019 zfit

from . import ztf  # legacy
from . import ztf as z
from .settings import ztypes

import tensorflow as tf

tf.get_variable_scope().set_use_resource(True)
tf.get_variable_scope().set_dtype(ztypes.float)

from . import constraint, pdf, minimize, loss, core, data, func
from .core.parameter import Parameter, ComposedParameter, ComplexParameter, convert_to_parameter
from .core.limits import Space, convert_to_space, supports
from .core.data import Data

from .settings import run

# EOF

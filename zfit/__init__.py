# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

__version__ = '0.2.0'

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2018, zfit"
__status__ = "Beta"

__author__ = "Jonas Eschle"
__maintainer__ = "zfit"
__email__ = 'zfit@physik.uzh.ch'
__credits__ = ["Jonas Eschle <jonas.eschle@cern.ch>",
               "Albert Puig <albert.puig@cern.ch",
               "Rafael Silva Coutinho <rafael.silva.coutinho@cern.ch>", ]

__all__ = ["ztf", "constraint", "pdf", "minimize", "loss", "core", "data", "func",
           "Parameter", "Space", "convert_to_space", "supports",
           "run", "settings"]

from . import ztf
from .settings import ztypes

import tensorflow as tf

tf.get_variable_scope().set_use_resource(True)
tf.get_variable_scope().set_dtype(ztypes.float)

from . import constraint, pdf, minimize, loss, core, data, func
from .core.parameter import Parameter
from .core.limits import Space, convert_to_space, supports

from .settings import run

# EOF

# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

__author__ = """zfit"""
__email__ = 'zfit@physik.uzh.ch'
__version__ = '0.0.0'

__all__ = ["ztf", "constraint", "pdf", "minimize", "loss", "core", "data", "func",
           "Parameter", "ComposedParameter", "ComplexParameter", "convert_to_parameter",
           "Space", "convert_to_space", "supports",
           "run", "settings"]

from . import ztf

import tensorflow as tf

tf.get_variable_scope().set_use_resource(True)

from . import constraint, pdf, minimize, loss, core, data, func
from .core.parameter import Parameter, ComposedParameter, ComplexParameter, convert_to_parameter
from .core.limits import Space, convert_to_space, supports
from .core.data import Data

from .settings import run

# EOF

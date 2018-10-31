# -*- coding: utf-8 -*-
"""Top-level package for zfit."""

__author__ = """zfit"""
__email__ = 'zfit@physik.uzh.ch'
__version__ = '0.0.0'


from . import pdf
from .core.parameter import FitParameter
from .core.loss import unbinned_nll
from .core.limits import Range, convert_to_range

# EOF

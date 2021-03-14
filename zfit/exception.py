#  Copyright (c) 2021 zfit
""".. todo:: Add module docstring."""
from .util.checks import NONE
from .util.exception import NameAlreadyTakenError, IllegalInGraphModeError, NormRangeNotImplementedError, \
    MultipleLimitsNotImplementedError, VectorizedLimitsNotImplementedError, AnalyticIntegralNotImplementedError, \
    AnalyticSamplingNotImplementedError, AnalyticNotImplementedError, FunctionNotImplementedError, \
    SpecificFunctionNotImplementedError, MinimizerSubclassingError, InitNotImplemented, MaximumIterationReached

__all__ = ['NameAlreadyTakenError', 'IllegalInGraphModeError', 'NormRangeNotImplementedError',
           'MultipleLimitsNotImplementedError', 'VectorizedLimitsNotImplementedError',
           'AnalyticNotImplementedError', 'FunctionNotImplementedError', 'AnalyticSamplingNotImplementedError',
           'AnalyticIntegralNotImplementedError', 'SpecificFunctionNotImplementedError', 'MinimizerSubclassingError',
           "MaximumIterationReached",
           'InitNotImplemented', 'NONE']

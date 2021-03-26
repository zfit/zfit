#  Copyright (c) 2021 zfit
""".. todo:: Add module docstring."""
from .util.checks import NONE
from .util.exception import (AnalyticIntegralNotImplementedError,
                             AnalyticNotImplementedError,
                             AnalyticSamplingNotImplementedError,
                             FunctionNotImplementedError,
                             IllegalInGraphModeError, InitNotImplemented,
                             MaximumIterationReached,
                             MinimizerSubclassingError,
                             MultipleLimitsNotImplementedError,
                             NameAlreadyTakenError,
                             NormRangeNotImplementedError,
                             SpecificFunctionNotImplementedError,
                             VectorizedLimitsNotImplementedError)

__all__ = ['NameAlreadyTakenError', 'IllegalInGraphModeError', 'NormRangeNotImplementedError',
           'MultipleLimitsNotImplementedError', 'VectorizedLimitsNotImplementedError',
           'AnalyticNotImplementedError', 'FunctionNotImplementedError', 'AnalyticSamplingNotImplementedError',
           'AnalyticIntegralNotImplementedError', 'SpecificFunctionNotImplementedError', 'MinimizerSubclassingError',
           "MaximumIterationReached",
           'InitNotImplemented', 'NONE']

#  Copyright (c) 2020 zfit
from .util.exception import NameAlreadyTakenError, IllegalInGraphModeError, NormRangeNotImplementedError, \
    MultipleLimitsNotImplementedError, VectorizedLimitsNotImplementedError, AnalyticIntegralNotImplementedError, \
    AnalyticSamplingNotImplementedError, AnalyticNotImplementedError, FunctionNotImplementedError, \
    SpecificFunctionNotImplementedError

__all__ = ['NameAlreadyTakenError', 'IllegalInGraphModeError', 'NormRangeNotImplementedError',
           'MultipleLimitsNotImplementedError', 'VectorizedLimitsNotImplementedError',
           'AnalyticNotImplementedError', 'FunctionNotImplementedError', 'AnalyticSamplingNotImplementedError',
           'AnalyticIntegralNotImplementedError', 'SpecificFunctionNotImplementedError']

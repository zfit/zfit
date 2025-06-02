#  Copyright (c) 2025 zfit
"""Exceptions that are raised by zfit.

Some are to provide a more specific error message, others are to steer the execution by raising an error that will be
caught in the right place.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

from .core.serialmixin import NumpyArrayNotSerializableError
from .util.checks import NONE
from .util.exception import (
    AnalyticGradientNotAvailable,
    AnalyticIntegralNotImplemented,
    AnalyticNotImplemented,
    AnalyticSamplingNotImplemented,
    BreakingAPIChangeError,
    FunctionNotImplemented,
    IllegalInGraphModeError,
    InitNotImplemented,
    LogicalUndefinedOperationError,
    MaximumIterationReached,
    MinimizerSubclassingError,
    MultipleLimitsNotImplemented,
    NormRangeNotImplemented,
    ParamNameNotUniqueError,
    ShapeIncompatibleError,
    SpecificFunctionNotImplemented,
    VectorizedLimitsNotImplemented,
)

__all__ = [
    "NONE",
    "AnalyticGradientNotAvailable",
    "AnalyticIntegralNotImplemented",
    "AnalyticNotImplemented",
    "AnalyticSamplingNotImplemented",
    "FunctionNotImplemented",
    "IllegalInGraphModeError",
    "InitNotImplemented",
    "InvalidNameError",
    "IpyoptPicklingError",
    "LogicalUndefinedOperationError",
    "MaximumIterationReached",
    "MinimizerSubclassingError",
    "MultipleLimitsNotImplemented",
    "NameAlreadyTakenError",
    "NormRangeNotImplemented",
    "NumpyArrayNotSerializableError",
    "ParamNameNotUniqueError",
    "ShapeIncompatibleError",
    "SpecificFunctionNotImplemented",
    "VectorizedLimitsNotImplemented",
]


class InvalidNameError(Exception):
    """Exception raised when a name is invalid."""


class NameAlreadyTakenError(Exception):
    def __init__(self, *_: object) -> None:
        raise BreakingAPIChangeError(
            msg="NameAlreadyTakenError has been removed and the behavior has substantially changed:"
            "parameters are now allowed to exist with the same as long as they are not within the same PDF/loss/func."
        )


class IpyoptPicklingError(TypeError):
    pass


class OutsideLimitsError(Exception):
    pass


class AutogradNotSupported(Exception):
    pass

#  Copyright (c) 2022 zfit

# TODO: improve errors of models. Generate more general error, inherit and use more specific?
import warnings


class PDFCompatibilityError(Exception):
    pass


class LogicalUndefinedOperationError(Exception):
    pass


class OperationNotAllowedError(Exception):
    pass


class ExtendedPDFError(Exception):
    pass


class AlreadyExtendedPDFError(ExtendedPDFError):
    pass


class NotExtendedPDFError(ExtendedPDFError):
    pass


class ConversionError(Exception):
    pass


class SubclassingError(Exception):
    pass


class BasePDFSubclassingError(SubclassingError):
    pass


class MinimizerSubclassingError(SubclassingError):
    pass


class IntentionAmbiguousError(Exception):
    pass


class UnderdefinedError(IntentionAmbiguousError):
    pass


class LimitsUnderdefinedError(UnderdefinedError):
    pass


class NormRangeUnderdefinedError(UnderdefinedError):
    pass


class OverdefinedError(IntentionAmbiguousError):
    pass


class LimitsOverdefinedError(OverdefinedError):
    pass


class CoordinatesUnderdefinedError(UnderdefinedError):
    pass


class AxesAmbiguousError(IntentionAmbiguousError):
    pass


class NotSpecifiedError(Exception):
    pass


class LimitsNotSpecifiedError(NotSpecifiedError):
    pass


class NormRangeNotSpecifiedError(NotSpecifiedError):
    pass


class AxesNotSpecifiedError(NotSpecifiedError):
    pass


class ObsNotSpecifiedError(NotSpecifiedError):
    pass


# Parameter Errors
class NameAlreadyTakenError(Exception):
    pass


# Operation errors
class IncompatibleError(Exception):
    pass


class ShapeIncompatibleError(IncompatibleError):
    pass


class ObsIncompatibleError(IncompatibleError):
    pass


class AxesIncompatibleError(IncompatibleError):
    pass


class CoordinatesIncompatibleError(IncompatibleError):
    pass


class SpaceIncompatibleError(IncompatibleError):
    pass


class LimitsIncompatibleError(IncompatibleError):
    pass


class NumberOfEventsIncompatibleError(ShapeIncompatibleError):
    pass


class InvalidLimitSubspaceError(Exception):
    pass


class ModelIncompatibleError(IncompatibleError):
    pass


# Data errors
class WeightsNotImplementedError(Exception):
    pass


class DataIsBatchedError(Exception):
    pass


# Parameter errors
class ParameterNotIndependentError(Exception):
    pass


# Minimizer errors


class NotMinimizedError(Exception):
    pass


# Runtime Errors


class IllegalInGraphModeError(Exception):
    pass


class CannotConvertToNumpyError(Exception):
    pass


# Baseclass to steer execution
class ZfitNotImplementedError(NotImplementedError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        if type(self) == ZfitNotImplementedError:
            warnings.warn(
                "Prefer to use a more specific subclass. See in `zfit.exceptions`"
            )


class FunctionNotImplemented(ZfitNotImplementedError):
    """Any function, e.g. in a BaseModel, that not implemented and a fallback should be called.

    Preferably use more specific exceptions
    """


class StandardControlFlow(Exception):
    """An exception that inherits from this class will be regarded as part of the standard control flow and not as an
    Error.

    For example, if a function raises that values are NaN, this is often intercepted on purpose.
    """


class SpecificFunctionNotImplemented(FunctionNotImplemented):
    """If a specific function, e.g. by the user is not implemented."""


class MinimizeNotImplemented(FunctionNotImplemented):
    """The `minimize` function of a minimizer is not implemented."""


class MinimizeStepNotImplemented(FunctionNotImplemented):
    """The `step` function of a minimizer is not implemented."""


class AnalyticNotImplemented(ZfitNotImplementedError):
    """General exception if an analytic way is not implemented."""


class AnalyticIntegralNotImplemented(AnalyticNotImplemented):
    """If an analytic integral is not provided."""

    pass


class AnalyticSamplingNotImplemented(AnalyticNotImplemented):
    """If analytic sampling from a distribution is not possible."""

    pass


# PDF class internal handling errors
class NormNotImplemented(StandardControlFlow):
    """Indicates that a function does not support the normalization range argument `norm_range`."""

    pass


NormRangeNotImplemented = NormNotImplemented  # legacy


class MultipleLimitsNotImplemented(StandardControlFlow):
    """Indicates that a function does not support several limits in a :py:class:`~zfit.Space`."""

    pass


class InitNotImplemented(StandardControlFlow):
    """Indicates that a minimize method does not support a FitResult instead of a loss."""

    pass


class VectorizedLimitsNotImplemented(StandardControlFlow):
    """Indicates that a function does not support vectorized (n_events > 1) limits in a :py:class:`~zfit.Space`."""

    pass


class DerivativeCalculationError(ValueError):
    pass


# Developer verbose messages


class WorkInProgressError(Exception):
    """Only for developing purpose!

    Does not serve as a 'real' Exception.
    """

    pass


class BreakingAPIChangeError(Exception):
    def __init__(self, msg, *args: object) -> None:
        default_msg = (
            "This item has been removed due to an API change. Instruction to update:\n"
            ""
        )
        msg = default_msg + str(msg)
        super().__init__(msg, *args)


class BehaviorUnderDiscussion(Exception):
    def __init__(self, msg, *args: object) -> None:
        default_msg = (
            "The behavior of the following is currently under discussion and ideas are well needed. "
            "Please open an issue at https://github.com/zfit/zfit/issues with your opinion about this.\n"
            ""
        )
        msg = default_msg + str(msg)
        super().__init__(msg, *args)


class MaximumIterationReached(StandardControlFlow):
    pass

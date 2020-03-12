#  Copyright (c) 2020 zfit

# TODO: improve errors of models. Generate more general error, inherit and use more specific?


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


class IntentionAmbiguousError(Exception):
    pass


class UnderdefinedError(IntentionAmbiguousError):
    pass


class LimitsUnderdefinedError(UnderdefinedError):
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


# PDF class internal handling errors
class NormRangeNotImplementedError(Exception):
    """Indicates that a function does not support the normalization range argument `norm_range`."""
    pass


class MultipleLimitsNotImplementedError(Exception):
    """Indicates that a function does not support several limits in a :py:class:`~zfit.Space`."""
    pass


# Developer verbose messages

class WorkInProgressError(Exception):
    """Only for developing purpose! Does not serve as a 'real' Exception."""
    pass


class BreakingAPIChangeError(Exception):
    pass


class BehaviorUnderDiscussion(Exception):

    def __init__(self, msg, *args: object) -> None:
        default_msg = ("The behavior of the following is currently under discussion and ideas are well needed. "
                       "Please open an issue at https://github.com/zfit/zfit/issues with your opinion about this.\n"
                       "")
        msg = default_msg + str(msg)
        super().__init__(msg, *args)

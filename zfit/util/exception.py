#  Copyright (c) 2019 zfit

# TODO: improve errors of models. Generate more general error, inherit and use more specific?


class PDFCompatibilityError(Exception):
    pass


class LogicalUndefinedOperationError(Exception):
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


class IntentionNotUnambiguousError(Exception):
    pass


class UnderdefinedError(IntentionNotUnambiguousError):
    pass


class LimitsUnderdefinedError(UnderdefinedError):
    pass


class OverdefinedError(IntentionNotUnambiguousError):
    pass


class LimitsOverdefinedError(OverdefinedError):
    pass


class AxesNotUnambiguousError(IntentionNotUnambiguousError):
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

class SpaceIncompatibleError(IncompatibleError):
    pass


class LimitsIncompatibleError(IncompatibleError):
    pass


class ModelIncompatibleError(IncompatibleError):
    pass


# Data errors
class WeightsNotImplementedError(Exception):
    pass


# Minimizer errors

class NotMinimizedError(Exception):
    pass


# Runtime Errors

class NoSessionSpecifiedError(Exception):
    pass


# PDF class internal handling errors
class NormRangeNotImplementedError(Exception):
    """Indicates that a function does not support the normalization range argument `norm_range`."""
    pass


class MultipleLimitsNotImplementedError(Exception):
    """Indicates that a function does not support several limits in a :py:class:`~zfit.Space`."""
    pass


# Developer verbose messages

class DueToLazynessNotImplementedError(Exception):
    """Only for developing purpose! Does not serve as a 'real' Exception."""
    pass

# TODO: improve errors of models. Generate more general error, inherit and use more specific?


class PDFCompatibilityError(Exception):
    pass


class LogicalUndefinedOperationError(ValueError):
    pass


class ExtendedPDFError(Exception):
    pass


class AlreadyExtendedPDFError(ExtendedPDFError):
    pass


class ConversionError(Exception):
    pass


class SubclassingError(Exception):
    pass


class BasePDFSubclassingError(SubclassingError):
    pass


class IntentionNotUnambiguousError(Exception):
    pass


class AxesNotUnambiguousError(IntentionNotUnambiguousError):
    pass


# Minimizer errors

class NotMinimizedError(Exception):
    pass


# PDF class internal handling errors
class NormRangeNotImplementedError(Exception):
    """Indicates that a function does not support the normalization range argument `norm_range`."""
    pass


class MultipleLimitsNotImplementedError(Exception):
    """Indicates that a function does not support several limits in a `Range`."""
    pass

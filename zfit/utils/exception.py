from __future__ import print_function, division, absolute_import


# TODO: improve errors of pdfs. Generate more general error, inherit and use more specific?


class PDFCompatibilityError(Exception):
    pass


class ExtendedPDFError(Exception):
    pass


class NormRangeNotImplementedError(Exception):
    """Indicates that a function does not support the normalization range argument `norm_range`."""
    pass

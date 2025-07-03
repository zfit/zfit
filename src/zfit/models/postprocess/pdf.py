"""Post-processing PDF functors for modifying PDF outputs.

These functors wrap other PDFs and modify their output in some way,
such as clipping values to ensure they are within certain bounds.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import typing
from typing import Literal

import pydantic.v1 as pydantic

import zfit.z.numpy as znp
from zfit._interfaces import ZfitPDF

from ...core.serialmixin import SerializableMixin
from ...core.space import supports
from ...serialization.pdfrepr import BasePDFRepr
from ...util import ztyping
from ...util.exception import SpecificFunctionNotImplemented
from ...util.ztyping import ExtendedInputType, NormInputType
from ..basefunctor import FunctorPDFRepr
from ..functor import BaseFunctor


class ClipPDF(BaseFunctor, SerializableMixin):
    """A functor that clips the output of a PDF to ensure it doesn't produce negative or NaN values.

    This is useful for PDFs that can produce negative values (e.g., KDE with negative weights) or
    numerical instabilities that lead to NaN values. The clipping operation uses znp.maximum and
    znp.minimum to ensure the output is within the specified bounds.

    Args:
        pdf: The PDF to clip
        lower: The minimum value to clip the output to. Default is 1e-100.
        upper: The maximum value to clip the output to. Default is None (no upper limit).
        obs: Observables of the PDF. If not given, taken from the wrapped PDF.
        extended: Whether the PDF is extended. If not given, taken from the wrapped PDF.
        norm: Normalization range. If not given, taken from the wrapped PDF.
        name: Name of the PDF
    """

    def __init__(
        self,
        pdf: ZfitPDF,
        lower: float = 1e-100,
        upper: float = None,
        obs: ztyping.ObsTypeInput = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "ClipPDF",
        **kwargs,
    ):
        self.pdf = pdf
        self.lower = znp.asarray(lower, dtype=pdf.dtype) if lower is not None else None
        self.upper = znp.asarray(upper, dtype=pdf.dtype) if upper is not None else None

        # Use the wrapped PDF's properties if not explicitly provided
        if obs is None:
            obs = pdf.obs
        if extended is None:
            extended = pdf.is_extended
        if norm is None:
            norm = pdf.norm

        super().__init__(pdfs=[pdf], obs=obs, extended=extended, norm=norm, name=name, **kwargs)

    def _clip_value(self, value):
        """Apply clipping to a value if bounds are specified."""
        if self.lower is not None:
            value = znp.maximum(value, self.lower)
        if self.upper is not None:
            value = znp.minimum(value, self.upper)
        return value

    @supports()
    def _unnormalized_pdf(self, x, params):
        """Return the clipped unnormalized PDF value."""
        value = self.pdf.pdf(x, norm=False)
        return self._clip_value(value)

    @supports(norm=True, multiple_limits=True)
    def _pdf(self, x, norm, params):
        """Return the clipped PDF value."""
        value = self.pdf.pdf(x, norm=norm)
        return self._clip_value(value)

    @supports(norm=True, multiple_limits=True)
    def _ext_pdf(self, x, norm, params):
        """Return the clipped extended PDF value."""
        if not self.pdf.is_extended:
            msg = f"PDF {self.pdf} is not extended but extended PDF was called."
            raise SpecificFunctionNotImplemented(msg)
        value = self.pdf.ext_pdf(x, norm=norm)
        return self._clip_value(value)

    @supports()
    def _ext_integrate(self, limits, norm, options):
        """Delegate extended integration to the wrapped PDF."""
        return self.pdf.ext_integrate(limits=limits, norm=norm, options=options)


class ClipPDFRepr(FunctorPDFRepr):
    _implementation = ClipPDF
    hs3_type: Literal["ClipPDF"] = pydantic.Field("ClipPDF", alias="type")

    pdf: BasePDFRepr = pydantic.Field(alias="pdf")
    lower: float | None = pydantic.Field(alias="lower")
    upper: float | None = pydantic.Field(alias="upper")
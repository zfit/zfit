"""Post-processing PDF functors for modifying PDF outputs.

These functors wrap other PDFs and modify their output in some way,
such as clipping values to ensure they are within certain bounds.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from typing import Literal

import pydantic.v1 as pydantic

import zfit.z.numpy as znp

from .._interfaces import ZfitPDF
from ..core.serialmixin import SerializableMixin
from ..core.space import supports
from ..serialization import Serializer  # noqa: F401
from ..util import ztyping
from ..util.exception import SpecificFunctionNotImplemented
from ..util.ztyping import ExtendedInputType, NormInputType
from .basefunctor import FunctorPDFRepr
from .functor import BaseFunctor

__all__ = ["PositivePDF"]


class PositivePDF(BaseFunctor, SerializableMixin):
    """A functor that ensures the output of a PDF is always positive by clipping values below epsilon.

    This is useful for PDFs that can produce negative values (e.g., KDE with negative weights) or
    numerical instabilities that lead to values very close to zero or NaN. The functor uses znp.maximum
    to ensure the output is always at least epsilon, and also replaces any NaN values with epsilon.

    Args:
        pdf: The PDF to make positive
        epsilon: The minimum positive value for the PDF output. Default is 1e-100.
        obs: Observables of the PDF. If not given, taken from the wrapped PDF.
        extended: Whether the PDF is extended. If not given, taken from the wrapped PDF.
        norm: Normalization range. If not given, taken from the wrapped PDF.
        name: Name of the PDF
    """

    def __init__(
        self,
        pdf: ZfitPDF,
        epsilon: float = 1e-100,
        obs: ztyping.ObsTypeInput = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "PositivePDF",
        **kwargs,
    ):
        self.epsilon = znp.asarray(epsilon, dtype=pdf.dtype)

        # Use the wrapped PDF's properties if not explicitly provided
        if obs is None:
            obs = pdf.obs
        if extended is None:  # TODO: yield as integral?
            extended = pdf.get_yield()
        if norm is None:
            norm = pdf.norm

        super().__init__(pdfs=[pdf], obs=obs, extended=extended, norm=norm, name=name, **kwargs)

    def _ensure_positive(self, value):
        """Ensure the value is at least epsilon and handle NaN values."""
        # Ensure all values are at least epsilon
        return znp.maximum(value, self.epsilon)

    @supports(norm=False)  # we cannot normalize easily, we only know (potentially) the norm of the wrapped PDF
    def _pdf(self, x, norm, params):
        """Return the PDF value ensuring it's at least epsilon."""
        assert not norm
        value = self.pdfs[0].pdf(x, norm=norm, params=params)
        return self._ensure_positive(value)

    @supports(norm=False)  # we cannot normalize easily, we only know (potentially) the norm of the wrapped PDF
    def _ext_pdf(self, x, norm, params):
        """Return the extended PDF value ensuring it's at least epsilon."""
        assert not norm
        if not self.pdfs[0].is_extended:
            msg = f"PDF {self.pdfs[0]} is not extended but extended PDF was called."
            raise SpecificFunctionNotImplemented(msg)
        value = self.pdfs[0].ext_pdf(x, norm=norm, params=params)
        return self._ensure_positive(value)


class PositivePDFRepr(FunctorPDFRepr):
    _implementation = PositivePDF
    hs3_type: Literal["PositivePDF"] = pydantic.Field("PositivePDF", alias="type")

    epsilon: float = pydantic.Field(alias="epsilon")

#  Copyright (c) 2023 zfit
from __future__ import annotations

import numpy as np
from uhi.typing.plottable import PlottableHistogram

import zfit.z.numpy as znp
from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.interfaces import ZfitBinnedData
from ..core.space import supports
from ..util import ztyping
from ..util.exception import SpecificFunctionNotImplemented


class HistogramPDF(BaseBinnedPDFV1):
    def __init__(
        self,
        data: ztyping.BinnedDataInputType,
        extended: ztyping.ExtendedInputType | None = None,
        norm: ztyping.NormInputType | None = None,
        name: str = "HistogramPDF",
    ) -> None:
        """Binned PDF resembling a histogram.

        Simple histogram PDF that can be used to model a histogram as a PDF.


        Args:
            data: Histogram to be used as PDF.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
               |@doc:pdf.init.extended.auto| If ``True``,
               the PDF will be extended automatically if the PDF is extended
               using the total number of events in the histogram.
               This is the default. |@docend:pdf.init.extended.auto|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
        """
        if extended is None:
            extended = True
        if not isinstance(data, ZfitBinnedData):
            if isinstance(data, PlottableHistogram):
                from zfit._data.binneddatav1 import BinnedData

                data = BinnedData.from_hist(data)
            else:
                raise TypeError(
                    "data must be of type PlottableHistogram (UHI) or ZfitBinnedData"
                )

        params = {}
        if extended is True:
            self._automatically_extended = True
            extended = znp.sum(data.values())
        else:
            self._automatically_extended = False
        super().__init__(
            obs=data.space, extended=extended, norm=norm, params=params, name=name
        )
        self._data = data

    @supports(norm="space")
    def _ext_pdf(self, x, norm):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm="space")
    def _pdf(self, x, norm):
        counts = self._rel_counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm="space")
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._data.values()
        return values

    @supports(norm="space")
    def _rel_counts(self, x, norm=None):
        values = self._data.values()
        return values / znp.sum(values)

#  Copyright (c) 2021 zfit
import numpy as np
from uhi.typing.plottable import PlottableHistogram
import zfit.z.numpy as znp

from ..core.space import supports
from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.interfaces import ZfitBinnedData
from ..util.exception import SpecificFunctionNotImplemented


class HistogramPDF(BaseBinnedPDFV1):

    def __init__(self, data, extended=None, norm=None, name="BinnedTemplatePDF"):
        if not isinstance(data, ZfitBinnedData):
            if isinstance(data, PlottableHistogram):
                from zfit._data.binneddatav1 import BinnedData
                data = BinnedData.from_hist(data)
            else:
                raise TypeError("data must be of type PlottableHistogram (UHI) or ZfitBinnedData")

        params = {}
        if extended is True:
            self._automatically_extended = True
            extended = znp.sum(data.values())
        else:
            self._automatically_extended = False
        super().__init__(obs=data.space, extended=extended, norm=norm, params=params, name=name)
        self._data = data

    @supports(norm='space')
    def _ext_pdf(self, x, norm):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm='space')
    def _pdf(self, x, norm):
        counts = self._rel_counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm='space')
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._data.values()
        return values

    @supports(norm='space')
    def _rel_counts(self, x, norm=None):
        values = self._data.values()
        return values / znp.sum(values)

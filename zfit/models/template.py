#  Copyright (c) 2020 zfit
from ..core.binnedpdf import BaseBinnedPDF
from ..util.exception import WorkInProgressError


class BinnedTemplatePDF(BaseBinnedPDF):

    def __init__(self, data, name="BinnedTemplatePDF"):
        obs = data.space
        super().__init__(obs=obs, name=name, params={})

        self._data = data

    def _ext_pdf(self, x, norm_range):
        return self._data.get_counts()

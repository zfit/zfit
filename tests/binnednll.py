#  Copyright (c) 2020 zfit

import boost_histogram as bh
import numpy as np

import zfit
from zfit.core.binneddata import BinnedData
from zfit.core.binning import RectBinning
from zfit.models.binned_functor import BinnedSumPDF
from zfit.models.template import BinnedTemplatePDF


def test_binned_nll_simple():
    counts = np.random.uniform(high=1, size=(10, 20))  # generate counts
    counts2 = np.random.normal(loc=5, size=(10, 20))
    counts3 = np.linspace(0, 10, num=10)[:, None] * np.linspace(0, 5, num=20)[None, :]
    binnings = [bh.axis.Regular(10, 0, 10), bh.axis.Regular(20, -10, 30)]
    binning = RectBinning(binnings=binnings)
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    mc1 = BinnedData.from_numpy(obs=obs, counts=counts, w2error=10)
    mc2 = BinnedData.from_numpy(obs=obs, counts=counts2, w2error=10)
    mc3 = BinnedData.from_numpy(obs=obs, counts=counts3, w2error=10)

    observed_data = BinnedData.from_numpy(obs=obs, counts=counts + counts2 + counts3, w2error=10)

    pdf = BinnedTemplatePDF(data=mc1)
    pdf2 = BinnedTemplatePDF(data=mc2)
    pdf3 = BinnedTemplatePDF(data=mc3)
    pdf.set_yield(np.sum(counts))
    pdf2.set_yield(np.sum(counts2))
    pdf3.set_yield(np.sum(counts3))
    # assert len(pdf.ext_pdf(None)) > 0
    pdf_sum = BinnedSumPDF(pdfs=[pdf, pdf2, pdf3], obs=obs)

    nll = zfit.loss.ExtendedBinnedNLL(pdf_sum, data=observed_data)
    nll.value()

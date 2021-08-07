#  Copyright (c) 2021 zfit

import boost_histogram as bh
import hist
import numpy as np

import zfit
import zfit.z.numpy as znp
from zfit.core.binneddata import BinnedData
from zfit.models.binned_functor import BinnedSumPDF
from zfit.models.template import BinnedTemplatePDF


def test_binned_nll_simple():
    zfit.run.set_graph_mode(False)
    counts = np.random.uniform(high=1, size=(10, 20))  # generate counts
    counts2 = np.random.normal(loc=5, size=(10, 20))
    counts3 = np.linspace(0, 10, num=10)[:, None] * np.linspace(0, 5, num=20)[None, :]
    binning = [hist.axis.Regular(10, 0, 10, name='obs1'), hist.axis.Regular(20, -10, 30, name='obs2')]
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    mc1 = BinnedData.from_tensor(space=obs, values=counts, variances=znp.ones_like(counts) * 1.3)
    mc2 = BinnedData.from_tensor(obs, counts2)
    mc3 = BinnedData.from_tensor(obs, counts3)
    sum_counts = counts + counts2 + counts3
    observed_data = BinnedData.from_tensor(space=obs, values=sum_counts, variances=(sum_counts + 0.5) ** 2)

    pdf = BinnedTemplatePDF(data=mc1)
    pdf2 = BinnedTemplatePDF(data=mc2)
    pdf3 = BinnedTemplatePDF(data=mc3)
    pdf._set_yield(np.sum(counts))
    pdf2._set_yield(np.sum(counts2))
    pdf3._set_yield(np.sum(counts3))
    # assert len(pdf.ext_pdf(None)) > 0
    pdf_sum = BinnedSumPDF(pdfs=[pdf, pdf2, pdf3], obs=obs)

    nll = zfit.loss.ExtendedBinnedNLL(pdf_sum, data=observed_data)
    nll.value()

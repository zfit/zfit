#  Copyright (c) 2021 zfit
import numpy as np
import pytest

import zfit.z.numpy as znp
import zfit
from zfit.data import BinnedData


def test_sum_histogram_pdf():
    bins1 = 5
    bins2 = 7
    counts = znp.random.uniform(high=1, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=5, size=(bins1, bins2))
    counts3 = znp.linspace(0, 10, num=bins1)[:, None] * znp.linspace(0, 5, num=bins2)[None, :]
    binnings = [zfit.binned.RegularBinning(bins1, 0, 10, name='obs1'),
                zfit.binned.RegularBinning(7, -10, bins2, name='obs2')]
    binning = binnings
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    data = BinnedData.from_tensor(space=obs, values=counts, variances=znp.ones_like(counts) * 1.3)
    data2 = BinnedData.from_tensor(obs, counts2)
    data3 = BinnedData.from_tensor(obs, counts3)

    pdf = zfit.pdf.HistogramPDF(data=data, extended=znp.sum(counts))
    pdf2 = zfit.pdf.HistogramPDF(data=data2, extended=znp.sum(counts2))
    pdf3 = zfit.pdf.HistogramPDF(data=data3, extended=znp.sum(counts3))
    assert len(pdf.ext_pdf(data)) > 0
    pdf_sum = zfit.pdf.BinnedSumPDF(pdfs=[pdf, pdf2, pdf3], obs=obs)

    probs = pdf_sum.counts(data)
    true_sum_counts = counts + counts2 + counts3
    np.testing.assert_allclose(true_sum_counts, probs)
    nsamples = 100_000_000
    sample = pdf_sum.sample(n=nsamples)
    np.testing.assert_allclose(true_sum_counts, sample.values() / nsamples * pdf_sum.get_yield(), rtol=0.03)

    # integrate
    true_integral = znp.sum(true_sum_counts)
    integral = pdf_sum.ext_integrate(limits=obs)
    assert pytest.approx(float(true_integral)) == float(integral)

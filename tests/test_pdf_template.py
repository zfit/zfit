#  Copyright (c) 2021 zfit

import boost_histogram as bh
import numpy as np
import progressbar
import pytest

import zfit
from zfit._loss.binnedloss import ExtendedBinnedNLL
from zfit.core.binneddata import BinnedData
from zfit.core.binning import RectBinning
from zfit.models.binned_functor import BinnedSumPDF
from zfit.models.template import BinnedTemplatePDF


def test_binned_template_pdf():
    bins1 = 5
    bins2 = 7
    counts = np.random.uniform(high=1, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=5, size=(bins1, bins2))
    counts3 = np.linspace(0, 10, num=bins1)[:, None] * np.linspace(0, 5, num=bins2)[None, :]
    binnings = [bh.axis.Regular(bins1, 0, 10), bh.axis.Regular(7, -10, bins2)]
    binning = RectBinning(binnings=binnings)
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    data = BinnedData.from_numpy(obs=obs, counts=counts, w2error=10)
    data2 = BinnedData.from_numpy(obs=obs, counts=counts2, w2error=10)
    data3 = BinnedData.from_numpy(obs=obs, counts=counts3, w2error=10)

    pdf = BinnedTemplatePDF(data=data)
    pdf2 = BinnedTemplatePDF(data=data2)
    pdf3 = BinnedTemplatePDF(data=data3)
    pdf.set_yield(np.sum(counts))
    pdf2.set_yield(np.sum(counts2))
    pdf3.set_yield(np.sum(counts3))
    assert len(pdf.ext_pdf(None)) > 0
    pdf_sum = BinnedSumPDF(pdfs=[pdf, pdf2, pdf3], obs=obs)

    probs = pdf_sum.ext_pdf(None)
    np.testing.assert_allclose(counts + counts2 + counts3, probs)

    # import matplotlib.pyplot as plt
    # plt.imshow(probs)
    # plt.imshow(counts2)
    # plt.show()
    # assert len(pdf.pdf(None, obs)) > 0


def test_binned_template_pdf_bbfull():
    bins1 = 100
    bins2 = 10

    counts1 = np.random.uniform(high=150, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=50, size=(bins1, bins2))
    counts3 = np.linspace(10, 100, num=bins1)[:, None] * np.linspace(10, 500, num=bins2)[None, :]
    binnings = [bh.axis.Regular(bins1, 0, 10), bh.axis.Regular(7, -10, bins2)]
    binning = RectBinning(binnings=binnings)
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)
    mc1 = BinnedData.from_numpy(obs=obs, counts=counts1, w2error=1)
    mc2 = BinnedData.from_numpy(obs=obs, counts=counts2, w2error=1)
    mc3 = BinnedData.from_numpy(obs=obs, counts=counts3, w2error=1)
    counts_mc = counts1 + counts2 + counts3

    counts1_data = np.random.uniform(high=150, size=(bins1, bins2))  # generate counts
    counts2_data = np.random.normal(loc=50, size=(bins1, bins2))
    counts3_data = np.linspace(10, 100, num=bins1)[:, None] * np.linspace(20, 490, num=bins2)[None, :]
    counts_data = counts1_data + counts2_data + counts3_data
    counts_data *= 1.1
    data = BinnedData.from_numpy(obs=obs, counts=counts_data, w2error=1)

    pdf1 = BinnedTemplatePDF(data=mc1)
    pdf2 = BinnedTemplatePDF(data=mc2)
    pdf3 = BinnedTemplatePDF(data=mc3)
    pdf1.set_yield(np.sum(counts1))
    pdf2.set_yield(np.sum(counts2))
    pdf3.set_yield(np.sum(counts3))
    assert len(pdf1.ext_pdf(None)) > 0
    pdf_sum = BinnedSumPDF(pdfs=[pdf1, pdf2, pdf3], obs=obs)
    counts1_flat = np.reshape(counts1, -1)
    constraints1 = zfit.constraint.GaussianConstraint(pdf1.params.values(), observation=np.ones_like(counts1_flat),
                                                      uncertainty=np.sqrt(
                                                          counts1_flat) / counts1_flat)
    counts2_flat = np.reshape(counts2, -1)
    constraints2 = zfit.constraint.GaussianConstraint(pdf2.params.values(), observation=np.ones_like(counts2_flat),
                                                      uncertainty=np.sqrt(
                                                          counts2_flat) / counts2_flat)
    counts3_flat = np.reshape(counts3, -1)
    constraints3 = zfit.constraint.GaussianConstraint(pdf3.params.values(), observation=np.ones_like(counts3_flat),
                                                      uncertainty=np.sqrt(
                                                          counts3_flat) / counts3_flat)
    # constraints2 = zfit.constraint.PoissonConstraint(pdf2.params.values(), np.reshape(counts2, -1))
    # constraints3 = zfit.constraint.PoissonConstraint(pdf3.params.values(), np.reshape(counts3, -1))
    loss = ExtendedBinnedNLL(pdf_sum, data, constraints=[
        constraints1,
        constraints2,
        constraints3
    ])
    for i in progressbar.progressbar(range(1000000)):
        loss.value()
        loss.gradients()
    minimizer = zfit.minimize.Minuit(verbosity=8, use_minuit_grad=False)
    minimizer.minimize(loss)

    probs = pdf_sum.ext_pdf(None)
    assert np.all(counts_data > probs)
    assert np.all(probs > counts_mc)
    # np.testing.assert_allclose(counts_data, probs, rtol=0.01)

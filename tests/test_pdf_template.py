#  Copyright (c) 2021 zfit

import hist
import mplhep
import numpy as np
import pytest
from hist.axestuple import NamedAxesTuple
from matplotlib import pyplot as plt

import zfit
import zfit.z.numpy as znp
from zfit._loss.binnedloss import ExtendedBinnedNLL
from zfit._data.binneddatav1 import BinnedDataV1
from zfit.models.binned_functor import BinnedSumPDFV1
from zfit.models.morphing import LinearMorphing
from zfit.models.template import BinnedTemplatePDFV1


def test_binned_template_pdf():
    bins1 = 5
    bins2 = 7
    counts = np.random.uniform(high=1, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=5, size=(bins1, bins2))
    counts3 = np.linspace(0, 10, num=bins1)[:, None] * np.linspace(0, 5, num=bins2)[None, :]
    binnings = [zfit.binned.Regular(bins1, 0, 10, name='obs1'), zfit.binned.Regular(7, -10, bins2, name='obs2')]
    binning = binnings
    axes = zfit.binned.Binning(binning)
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    data = BinnedDataV1.from_tensor(space=obs, values=counts, variances=znp.ones_like(counts) * 1.3)
    data2 = BinnedDataV1.from_tensor(obs, counts2)
    data3 = BinnedDataV1.from_tensor(obs, counts3)

    pdf = BinnedTemplatePDFV1(data=data, extended=np.sum(counts))
    pdf2 = BinnedTemplatePDFV1(data=data2, extended=np.sum(counts2))
    pdf3 = BinnedTemplatePDFV1(data=data3, extended=np.sum(counts3))
    assert len(pdf.ext_pdf(data)) > 0
    pdf_sum = BinnedSumPDFV1(pdfs=[pdf, pdf2, pdf3], obs=obs)

    probs = pdf_sum.counts(data)
    true_sum_counts = counts + counts2 + counts3
    np.testing.assert_allclose(true_sum_counts, probs)
    nsamples = 100_000_000
    sample = pdf_sum.sample(n=nsamples)
    np.testing.assert_allclose(true_sum_counts, sample.values() / nsamples * pdf_sum.get_yield(), rtol=0.03)

    # integrate
    true_integral = znp.sum(true_sum_counts * np.prod(axes.widths, axis=0))
    integral = pdf_sum.ext_integrate(limits=obs)
    assert pytest.approx(float(true_integral)) == float(integral)

    # import matplotlib.pyplot as plt
    # plt.imshow(probs)
    # plt.imshow(counts2)
    # plt.show()
    # assert len(pdf.pdf(None, obs)) > 0


def test_morphing_templates():
    bins1 = 10
    counts1 = np.random.uniform(70, high=100, size=bins1)  # generate counts
    counts = [counts1 - np.random.uniform(high=20, size=bins1), counts1,
              counts1 + np.random.uniform(high=20, size=bins1)]
    binning = zfit.binned.Regular(bins1, 0, 10, name='obs1')
    obs = zfit.Space(obs='obs1', binning=binning)
    datasets = [BinnedDataV1.from_tensor(obs, count) for count in counts]
    pdfs = [BinnedTemplatePDFV1(data=data, extended=np.sum(data.values())) for data in datasets]
    alpha = zfit.Parameter('alpha', 0, -5, 5)
    morph = LinearMorphing(alpha=alpha, hists=pdfs)
    np.testing.assert_allclose(morph.counts(), counts[1])
    alpha.set_value(1)
    np.testing.assert_allclose(morph.counts(), counts[2])
    alpha.set_value(-1)
    np.testing.assert_allclose(morph.counts(), counts[0])

    hists = []
    import matplotlib.cm as cm

    amin, amax = -2, 2
    n = 5
    for a in znp.linspace(amin, amax, n * 4 - 1):
        normed_a = (a - amin) / (amax - amin)
        color = cm.get_cmap('cool')(normed_a)
        alpha.set_value(a)
        histo = morph.ext_pdf(None)
        histo = BinnedDataV1.from_tensor(obs, histo).to_hist()
        if np.min((a - znp.linspace(amin, amax, n)) ** 2) < 0.001:
            mplhep.histplot(histo, label=f'alpha={a}', color=color)
        else:
            mplhep.histplot(histo, color=color)
        plt.legend()
    plt.show()


def test_morphing_templates2D():
    zfit.run.set_graph_mode(True)
    bins1 = 10
    bins2 = 7
    shape = (bins1, bins2)
    counts1 = np.random.uniform(70, high=100, size=shape)  # generate counts
    # counts1 = np.random.uniform(70, high=100, size=bins1)  # generate counts
    counts = [counts1 - np.random.uniform(high=20, size=shape), counts1,
              counts1 + np.random.uniform(high=20, size=shape)]
    binning1 = zfit.binned.Regular(bins1, 0, 10, name='obs1')
    binning2 = zfit.binned.Regular(bins2, 0, 10, name='obs2')
    obs1 = zfit.Space(obs='obs1', binning=binning1)
    obs2 = zfit.Space(obs='obs2', binning=binning2)
    obs = obs1 * obs2
    # obs._binning = NamedAxesTuple([binning1, binning2])  # TODO: does it work without this?
    datasets = [BinnedDataV1.from_tensor(obs, count) for count in counts]
    pdfs = [BinnedTemplatePDFV1(data=data, extended=np.sum(data.values())) for data in datasets]
    alpha = zfit.Parameter('alpha', 0, -5, 5)
    morph = LinearMorphing(alpha=alpha, hists=pdfs)
    np.testing.assert_allclose(morph.counts(), counts[1])
    alpha.set_value(1)
    np.testing.assert_allclose(morph.counts(), counts[2])
    alpha.set_value(-1)
    np.testing.assert_allclose(morph.counts(), counts[0])


def test_binned_template_pdf_bbfull():
    bins1 = 15
    bins2 = 10

    counts1 = np.random.uniform(high=150, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=50, size=(bins1, bins2))
    counts3 = np.linspace(10, 100, num=bins1)[:, None] * np.linspace(10, 500, num=bins2)[None, :]
    binnings = [zfit.binned.Regular(bins1, 0, 10, name='obs1'), zfit.binned.Regular(7, -10, bins2, name='obs2')]
    binning = binnings
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    mc1 = BinnedDataV1.from_tensor(space=obs, values=counts1, variances=znp.ones_like(counts1) * 1.3)
    mc2 = BinnedDataV1.from_tensor(obs, counts2)
    mc3 = BinnedDataV1.from_tensor(obs, counts3)

    counts_mc = counts1 + counts2 + counts3

    counts1_data = np.random.uniform(high=150, size=(bins1, bins2))  # generate counts
    counts2_data = np.random.normal(loc=50, size=(bins1, bins2))
    counts3_data = np.linspace(10, 100, num=bins1)[:, None] * np.linspace(20, 490, num=bins2)[None, :]
    counts_data = counts1_data + counts2_data + counts3_data
    counts_data *= 1.1
    data = BinnedDataV1.from_tensor(space=obs, values=counts_data)

    pdf1 = BinnedTemplatePDFV1(data=mc1)
    pdf2 = BinnedTemplatePDFV1(data=mc2)
    pdf3 = BinnedTemplatePDFV1(data=mc3)
    assert len(pdf1.counts()) > 0
    pdf_sum = BinnedSumPDFV1(pdfs=[pdf1, pdf2, pdf3], obs=obs)
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
    # for i in progressbar.progressbar(range(1000000)):
    #     loss.value()
    #     loss.gradients()
    minimizer = zfit.minimize.Minuit(verbosity=8, gradient=False)
    minimizer.minimize(loss)

    counts = pdf_sum.counts()
    assert np.all(counts_data > counts)
    assert np.all(counts > counts_mc)
    # np.testing.assert_allclose(counts_data, probs, rtol=0.01)

#  Copyright (c) 2021 zfit

import mplhep
import numpy as np
import numpy.testing
import pytest
import zfit
import zfit.z.numpy as znp
from matplotlib import pyplot as plt
from zfit._data.binneddatav1 import BinnedDataV1
from zfit._loss.binnedloss import ExtendedBinnedNLL
from zfit.models.binned_functor import BinnedSumPDFV1
from zfit.models.morphing import SplineMorphing
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
    true_integral = znp.sum(true_sum_counts)
    integral = pdf_sum.ext_integrate(limits=obs, )
    assert pytest.approx(float(true_integral)) == float(integral)

    # import matplotlib.pyplot as plt
    # plt.imshow(probs)
    # plt.imshow(counts2)
    # plt.show()
    # assert len(pdf.pdf(None, obs)) > 0

@pytest.mark.plots
@pytest.mark.parametrize('alphas', [None, [-0.7, -0.1, 0.5, 1.4]])
def test_morphing_templates(alphas):
    bins1 = 15
    irregular_str = 'irregular templates' if alphas is not None else ''

    counts1 = np.random.uniform(70, high=100, size=bins1)  # generate counts
    counts = [counts1 - np.random.uniform(high=20, size=bins1), counts1,
              counts1 + np.random.uniform(high=20, size=bins1)]
    if alphas is not None:
        counts.append(counts1 + np.random.uniform(high=5, size=bins1))
    binning = zfit.binned.Regular(bins1, 0, 10, name='obs1')
    obs = zfit.Space(obs='obs1', binning=binning)
    datasets = [BinnedDataV1.from_tensor(obs, count) for count in counts]
    pdfs = [BinnedTemplatePDFV1(data=data, extended=np.sum(data.values())) for data in datasets]
    if alphas is not None:
        pdfs = {a: p for a, p in zip(alphas, pdfs)}
    alpha = zfit.Parameter('alpha', 0, -5, 5)
    morph = SplineMorphing(alpha=alpha, hists=pdfs)
    if alphas is None:
        alphas = [-1, 0, 1]
    for i, a in enumerate(alphas):
        alpha.set_value(a)
        np.testing.assert_allclose(morph.counts(), counts[i])
        if len(alphas) > i + 1:
            alpha.set_value((a + alphas[i + 1]) / 2)
            max_dist = (counts[i] - counts[i + 1]) ** 2 + 5  # tolerance
            max_dist *= 1.1  # not strict, it can be a bit higher
            numpy.testing.assert_array_less((morph.counts() - counts[i]) ** 2, max_dist)
            numpy.testing.assert_array_less((morph.counts() - counts[i + 1]) ** 2, max_dist)

    hists = []
    import matplotlib.cm as cm

    amin, amax = -2, 2
    n = 5

    template_alphas = np.array(list(alphas))

    for do_3d in [True, False]:
        plt.figure()
        if do_3d:
            ax = plt.gcf().add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
        plotstyle = '3d plot' if do_3d else 'hist plot'
        plt.title(f"Morphing with splines {irregular_str} {plotstyle}")

        for a in list(znp.linspace(amin, amax, n * 2)) + list(template_alphas):
            normed_a = (a - amin) / (amax - amin) / 1.3  # 3 is a scaling factor
            color = cm.get_cmap('winter')(normed_a)
            alpha.set_value(a)
            histo = morph.ext_pdf(None)
            histo = BinnedDataV1.from_tensor(obs, histo)
            histo = histo.to_hist()
            values = histo.values()
            x = histo.axes.edges[0][:-1]
            y = np.broadcast_to(a, values.shape)
            z = values
            label = None
            if do_3d:
                ax.step(x, y, z, color=color, where='pre', label=label)
            else:
                if np.min((a - template_alphas) ** 2) < 0.0001:
                    label = f'alpha={a}'
                mplhep.histplot(histo, label=label, color=color)
        ax.set_xlabel('observable')
        ax.set_ylabel('alpha')
        if do_3d:
            ax.set_zlabel('ext_pdf')
        plt.legend()
        pytest.zfit_savefig()


@pytest.mark.parametrize('alphas', [None, [-0.7, -0.1, 0.5, 1.4]])
def test_morphing_templates2D(alphas):
    bins1 = 10
    bins2 = 7
    shape = (bins1, bins2)
    counts1 = np.random.uniform(70, high=100, size=shape)  # generate counts
    # counts1 = np.random.uniform(70, high=100, size=bins1)  # generate counts
    counts = [counts1 - np.random.uniform(high=20, size=shape), counts1,
              counts1 + np.random.uniform(high=20, size=shape)]
    if alphas is not None:
        counts.append(counts1 + np.random.uniform(high=5, size=shape))
    binning1 = zfit.binned.Variable(sorted(np.random.uniform(0, 10, size=bins1 + 1)), name='obs1')
    binning2 = zfit.binned.Regular(bins2, 0, 10, name='obs2')
    obs1 = zfit.Space(obs='obs1', binning=binning1)
    obs2 = zfit.Space(obs='obs2', binning=binning2)
    obs = obs1 * obs2
    datasets = [BinnedDataV1.from_tensor(obs, count) for count in counts]
    pdfs = [BinnedTemplatePDFV1(data=data, extended=np.sum(data.values())) for data in datasets]
    if alphas is not None:
        pdfs = {a: p for a, p in zip(alphas, pdfs)}
    alpha = zfit.Parameter('alpha', 0, -5, 5)
    morph = SplineMorphing(alpha=alpha, hists=pdfs)
    if alphas is None:
        alphas = [-1, 0, 1]
    for i, a in enumerate(alphas):
        alpha.set_value(a)
        np.testing.assert_allclose(morph.counts(), counts[i])
        assert pytest.approx(np.sum(counts[i])) == zfit.run(morph.get_yield())
        if len(alphas) > i + 1:
            alpha.set_value((a + alphas[i + 1]) / 2)
            max_dist = (counts[i] - counts[i + 1]) ** 2 + 5  # tolerance
            max_dist *= 1.1  # not strict, it can be a bit higher
            numpy.testing.assert_array_less((morph.counts() - counts[i]) ** 2, max_dist)
            numpy.testing.assert_array_less((morph.counts() - counts[i + 1]) ** 2, max_dist)


def test_binned_template_pdf_bbfull():
    bins1 = 15
    bins2 = 10

    counts1 = np.random.uniform(high=150, size=(bins1, bins2))  # generate counts
    counts2 = np.random.normal(loc=50, size=(bins1, bins2))
    counts3 = np.linspace(10, 100, num=bins1)[:, None] * np.linspace(10, 500, num=bins2)[None, :]
    binnings = [zfit.binned.Regular(bins1, 0, 10, name='obs1'), zfit.binned.Regular(bins2, -10, 7, name='obs2')]
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

    pdf1 = BinnedTemplatePDFV1(data=mc1, sysshape=True)
    pdf2 = BinnedTemplatePDFV1(data=mc2, sysshape=True)
    pdf3 = BinnedTemplatePDFV1(data=mc3, sysshape=True)
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
    loss.value()
    loss.gradients()
    print('start minimization')
    minimizer = zfit.minimize.Minuit(verbosity=8, gradient=False)
    minimizer.minimize(loss)

    counts = pdf_sum.counts()
    np.testing.assert_array_less(counts_mc, counts_data)  # this is an assumption, if that is wrong, the test is flawed
    np.testing.assert_array_less(counts, counts_data)
    np.testing.assert_array_less(counts_mc, counts)
    # np.testing.assert_allclose(counts_data, probs, rtol=0.01)

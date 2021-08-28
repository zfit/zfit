#  Copyright (c) 2020 zfit
import time
import pytest

import hist
import mplhep
import numpy as np
from matplotlib import pyplot as plt

import zfit.pdf
import zfit.z.numpy as znp
from zfit.core.binnedpdf import BinnedFromUnbinned
from zfit.core.unbinnedpdf import SplinePDF


@pytest.mark.plots
def test_binned_from_unbinned():
    # zfit.run.set_graph_mode(False)
    n = 1004
    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 1, 0, 19)
    obs = zfit.Space('x', (-5, 10))
    n_testpoints = n
    x = znp.linspace(-5, 10, n_testpoints // 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    gauss.set_yield(n)

    axis = zfit.binned.Regular(150, -5, 10, name='x')
    obs_binned = zfit.Space('x', binning=[axis])
    gauss_binned = BinnedFromUnbinned(pdf=gauss, space=obs_binned, extended=n)
    values = gauss_binned.rel_counts(obs_binned)
    for _ in range(5):
        values = gauss_binned.counts(obs_binned)
    sample = gauss_binned.sample(n, limits=obs_binned)

    title = 'Comparison of binned gaussian and sample'
    plt.figure()
    plt.title(title)
    mplhep.histplot(sample.to_hist(), label='sampled binned')
    plt.plot(axis.centers, gauss_binned.counts(obs_binned), label='counts binned')
    plt.legend()
    pytest.zfit_savefig()
    sample_unbinned = sample.to_unbinned()

    spline_gauss = SplinePDF(gauss_binned, obs=obs)
    # spline_gauss.set_yield(n)  # HACK
    y = spline_gauss.ext_pdf(x)
    y_unbinned = spline_gauss.ext_pdf(sample_unbinned)
    y_true = gauss.ext_pdf(x)
    plt.figure()
    plt.title("Comparison of unbinned gauss to binned to interpolated")
    plt.plot(axis.centers, gauss_binned.ext_pdf(obs_binned), 'x', label='binned')
    plt.plot(x, y_true, label='original')
    plt.plot(x, y, '.', label='interpolated')
    plt.legend()
    pytest.zfit_savefig()
    np.testing.assert_allclose(y, y_true, atol=50)


def test_binned_from_unbinned_2D():
    # zfit.run.set_graph_mode(True)

    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 4, 0, 120)
    obsx = zfit.Space('x', (-5, 10))
    obsy = zfit.Space('y', (-50, 100))
    gaussx = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obsx)
    gaussy = zfit.pdf.Gauss(mu=mu, sigma=sigma * 20, obs=obsy)
    gauss2D = zfit.pdf.ProductPDF([gaussx, gaussy])

    axisx = zfit.binned.Variable(sorted(np.random.uniform(-5, 10, size=10)), name="x")
    # axisx = zfit.binned.Regular(10, -5, 10, name='x')
    axisy = zfit.binned.Regular(100, -50, 100, name='y')
    obs_binnedx = zfit.Space(['x', 'y'], binning=[axisx, axisy])
    obs_binnedy = zfit.Space('y', binning=[axisy])
    obs_binned = obs_binnedx
    # obs_binned = obs_binnedx * obs_binnedy

    gauss_binned = BinnedFromUnbinned(pdf=gauss2D, space=obs_binned, extended=100)
    values = gauss_binned.rel_counts(obs_binned)

    start = time.time()
    for _ in range(2):
        values = gauss_binned.rel_counts(obs_binned)
    print(f"Time needed: {time.time() - start}")

    n = 10000
    sample = gauss_binned.sample(n, limits=obs_binned)
    hist1 = sample.to_hist()
    mplhep.hist2dplot(hist1)
    plt.figure()
    mplhep.hist2dplot(gauss_binned.to_hist())
    # plt.plot(axisx.centers, values * n)
    plt.show()

#  Copyright (c) 2020 zfit
import time

import hist
import mplhep
from matplotlib import pyplot as plt

import zfit.pdf
import zfit.z.numpy as znp
from zfit.core.binnedpdf import BinnedFromUnbinned
from zfit.core.unbinnedpdf import SplinePDF


def test_binned_from_unbinned():
    # zfit.run.set_graph_mode(False)

    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 1, 0, 19)
    obs = zfit.Space('x', (-5, 10))
    x = znp.linspace(-5, 10, 100)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    axis = hist.axis.Regular(150, -5, 10, name='x')
    obs_binned = zfit.Space('x', binning=[axis])
    gauss_binned = BinnedFromUnbinned(pdf=gauss, space=obs_binned, extended=100)
    values = gauss_binned.pdf(None)
    start = time.time()
    for _ in range(5):
        values = gauss_binned.pdf(None)
    print(f"Time needed: {time.time() - start}")
    n = 100000
    sample = gauss_binned.sample(n, limits=obs_binned)
    mplhep.histplot(sample.to_hist(), label='sampled binned')
    plt.plot(axis.centers, values * n)
    plt.show()

    spline_gauss = SplinePDF(gauss_binned, obs=obs)
    spline_gauss.set_yield(n)  # HACK
    y = spline_gauss.ext_pdf(x)
    plt.figure()
    mplhep.histplot(sample.to_hist(), label='sampled binned')
    plt.plot(axis.centers, values * n, label='original')
    plt.plot(x, y * 100, '.', label='interpolated')
    plt.legend()
    plt.show()


def test_binned_from_unbinned_2D():
    # zfit.run.set_graph_mode(True)

    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 2, 0, 19)
    obsx = zfit.Space('x', (-5, 10))
    obsy = zfit.Space('y', (-50, 100))
    gaussx = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obsx)
    gaussy = zfit.pdf.Gauss(mu=mu, sigma=sigma * 20, obs=obsy)
    gauss2D = zfit.pdf.ProductPDF([gaussx, gaussy])

    axisx = hist.axis.Regular(10, -5, 10, name='x')
    axisy = hist.axis.Regular(100, -50, 100, name='y')
    obs_binnedx = zfit.Space(['x', 'y'], binning=[axisx, axisy])
    obs_binnedy = zfit.Space('y', binning=[axisy])
    obs_binned = obs_binnedx
    # obs_binned = obs_binnedx * obs_binnedy

    gauss_binned = BinnedFromUnbinned(pdf=gauss2D, space=obs_binned, extended=100)
    values = gauss_binned.counts(None)
    values = gauss_binned.pdf(None)
    values = gauss_binned.pdf(None)
    start = time.time()
    for _ in range(2):
        values = gauss_binned.pdf(None)
    print(f"Time needed: {time.time() - start}")

    n = 10000
    sample = gauss_binned.sample(n, limits=obs_binned)
    hist1 = sample.to_hist()
    mplhep.hist2dplot(hist1)
    # plt.plot(axisx.centers, values * n)
    plt.show()

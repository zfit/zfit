#  Copyright (c) 2020 zfit
import mplhep
import numpy as np
import pytest
from matplotlib import pyplot as plt

import zfit.pdf
import zfit.z.numpy as znp
from zfit.core.binnedpdf import BinnedFromUnbinned
from zfit.core.unbinnedpdf import SplinePDF


@pytest.mark.plots
def test_binned_from_unbinned_spline():
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
    # values = gauss_binned.rel_counts(obs_binned)

    sample = gauss_binned.sample(n, limits=obs_binned)

    title = 'Comparison of binned gaussian and sample'
    plt.figure()
    plt.title(title)
    mplhep.histplot(sample.to_hist(), label='sampled binned')
    plt.plot(axis.centers, gauss_binned.counts(obs_binned), label='counts binned')
    plt.legend()
    pytest.zfit_savefig()

    spline_gauss = SplinePDF(gauss_binned, obs=obs)
    # spline_gauss.set_yield(n)  # HACK
    y = spline_gauss.ext_pdf(x)
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
    n = 100000

    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 4, 0, 120)
    obsx = zfit.Space('x', (-5, 10))
    obsy = zfit.Space('y', (-50, 100))
    gaussx = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obsx)
    gaussy = zfit.pdf.Gauss(mu=mu, sigma=sigma * 20, obs=obsy)
    gauss2D = zfit.pdf.ProductPDF([gaussx, gaussy])

    normal = np.random.normal(float(mu), float(sigma), size=500)
    axisx = zfit.binned.Variable(np.concatenate([np.linspace(-5, 5, 43), np.linspace(5, 10, 30)[1:]], axis=0), name="x")
    axisy = zfit.binned.Regular(15, -50, 100, name='y')
    obs_binnedx = zfit.Space(['x'], binning=axisx)
    obs_binnedy = zfit.Space('y', binning=axisy)
    obs_binned = obs_binnedx * obs_binnedy

    gauss_binned = BinnedFromUnbinned(pdf=gauss2D, space=obs_binned, extended=n)
    values = gauss_binned.rel_counts(obs_binned)  # TODO: good test?

    # start = time.time()
    # for _ in range(2):
    #     values = gauss_binned.rel_counts(obs_binned)
    # print(f"Time needed: {time.time() - start}")

    sample = gauss_binned.sample(n, limits=obs_binned)
    hist_sampled = sample.to_hist()
    hist_pdf = gauss_binned.to_hist()
    max_error = hist_sampled.values() * 6 ** 2  # 6 sigma away
    np.testing.assert_array_less((hist_sampled.values() - hist_pdf.values()) ** 2, max_error)
    plt.figure()
    plt.title("Gauss 2D binned sampled.")
    mplhep.hist2dplot(hist_sampled)
    pytest.zfit_savefig()
    plt.figure()
    plt.title("Gauss 2D binned plot, irregular (x<4.5 larger bins than x>4.5) binning.")
    mplhep.hist2dplot(hist_pdf)
    pytest.zfit_savefig()

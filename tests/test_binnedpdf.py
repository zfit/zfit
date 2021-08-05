#  Copyright (c) 2020 zfit
import time

import hist
import mplhep
from matplotlib import pyplot as plt

import zfit.pdf
from zfit.core.binnedpdf import BinnedFromUnbinned


def test_binned_from_unbinned():
    # zfit.run.set_graph_mode(False)

    mu = zfit.Parameter('mu', 1, 0, 19)
    sigma = zfit.Parameter('sigma', 1, 0, 19)
    obs = zfit.Space('x', (-5, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    axis = hist.axis.Regular(150, -5, 10, name='x')
    obs_binned = zfit.Space('x', binning=[axis])
    gauss_binned = BinnedFromUnbinned(pdf=gauss, space=obs_binned, extended=100)
    values = gauss_binned.pdf(None, norm=False)
    start = time.time()
    values = gauss_binned.pdf(None, norm=False)
    print(f"Time needed: {time.time() - start}")
    n = 10000
    sample = gauss_binned.sample(n, limits=obs_binned)
    mplhep.histplot(sample.to_hist())
    plt.plot(axis.centers, values * n)
    plt.show()

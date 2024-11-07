#  Copyright (c) 2024 zfit
import numpy as np

import zfit


def test_to_binned_polynomial():

    low, high = 1, 10
    obs = zfit.Space('obs', low, high, binning=100)
    size_normal = 10_000
    data_normal_np = np.random.normal(size=size_normal, loc=5, scale=1)
    data_expo_np = np.random.exponential(size=size_normal, scale=20)
    data_tot_np = np.concatenate([data_normal_np, data_expo_np])
    data = zfit.Data(data_tot_np, obs=obs)

    sigma = zfit.Parameter('sigma', 1.)
    alpha = zfit.Parameter('alpha', 90., 0., 100.)
    c0 = zfit.Parameter('c0', 0.1)
    c1 = zfit.Parameter('c1', -0.026, -100., 100.)
    c2 = zfit.Parameter('c2', -0.056, -100., 100.)

    pdf_1 = zfit.pdf.GeneralizedGaussExpTail(
        obs=obs,
        mu=5.,
        sigmar=sigma,
        sigmal=sigma,
        alphar=alpha,
        alphal=alpha
    )
    pdf_2 = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2], coeff0=c0)
    pdf = zfit.pdf.BinnedSumPDF([pdf_1, pdf_2], [0.5], obs=obs)
    vals = pdf.pdf(data)  # make sure this doesn't crash and the spaces work in the jitted mode
    assert not np.any(np.isnan(vals.numpy()))

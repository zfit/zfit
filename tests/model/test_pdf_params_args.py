#  Copyright (c) 2024 zfit
import numpy as np
import pytest

import zfit
import zfit.z.numpy as znp


def test_pdf_params_args_unbinned():

    mu = zfit.Parameter('muparam', 1.2, -1., 3.)
    sigma = zfit.Parameter('sigmaparam', 1.3, 0.1, 10.)
    obs = zfit.Space('obs1', limits=(-10, 10))

    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    data = znp.array([1.1, 1.2, 1.3, 1.4, 1.5])

    assert np.argmax(gauss.pdf(data)) == 1
    assert np.argmax(gauss.pdf(data, params={'muparam': 1.4})) == 3
    with pytest.raises(ValueError):
        gauss.pdf(data, params={'muparam': 1.4, 'sigma': 1.5})

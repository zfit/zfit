#  Copyright (c) 2024 zfit
import pytest
import numpy as np
#from scipy.special import gamma as gamma_scipy

import zfit
import zfit.z.numpy as znp

def test_expmodgauss():
    mu_true = 2
    sigma_true = 1
    lambd_true = 0.5
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    lambd = zfit.Parameter("lambda", lambd_true)

    obs = zfit.Space(obs="obs", limits=(-15, 30))
    expmodgauss = zfit.pdf.ExpModGauss(mu=mu, sigma=sigma, lambd=lambd, obs=obs)

    samples = expmodgauss.sample(100000)["obs"]
    probabilities = expmodgauss.pdf(x=samples)

    assert znp.all(np.isfinite(probabilities))
    assert znp.all(probabilities>=0)
    assert pytest.approx(mu_true + 1 / lambd_true, rel=1e-2) == znp.mean(samples)
    assert pytest.approx(znp.sqrt(sigma_true ** 2 + 1 / lambd_true ** 2), rel=1e-2) == znp.std(samples)

#  Copyright (c) 2024 zfit
import pytest
import numpy as np
from scipy.special import gamma as gamma_scipy

import zfit
import zfit.z.numpy as znp


def test_generalizedgauss():
    mu_true = 2
    sigma_true = 1
    beta_true = 2.5
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    beta = zfit.Parameter("beta", beta_true)

    obs = zfit.Space(obs="obs", limits=(-15, 15))
    t = zfit.pdf.GeneralizedGauss(mu=mu, sigma=sigma, beta=beta, obs=obs)
    test_values = np.random.uniform(low=-15.0, high=15.0, size=1000)
    samples = t.sample(100_000)["obs"]
    probs = t.pdf(x=test_values)

    assert np.all(np.isfinite(probs))
    assert np.all(probs>=0)
    assert pytest.approx(2, rel=1e-2) == znp.mean(samples)
    assert pytest.approx(sigma * np.sqrt(gamma_scipy(3/2.5) / gamma_scipy(1/2.5)), rel=1e-2) == znp.std(samples)

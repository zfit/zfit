#  Copyright (c) 2024 zfit
import pytest
import numpy as np

import zfit
import zfit.z.numpy as znp


def test_studentt():
    N_true = 5
    mu_true = 2
    sigma_true = 1
    N = zfit.Parameter("N", N_true)
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)

    obs = zfit.Space(obs="obs", limits=(-50, 50))
    t = zfit.pdf.StudentT(ndof=N, mu=mu, sigma=sigma, obs=obs)
    test_values = np.random.uniform(low=-10.0, high=10.0, size=1000)
    samples = t.sample(100_000)["obs"]
    probs = t.pdf(x=test_values)

    assert np.all(np.isfinite(probs))
    assert np.all(probs>=0)
    assert pytest.approx(2, rel=1e-2) == znp.mean(samples)
    assert pytest.approx(np.sqrt(N_true / (N_true - 2)), rel=1e-2) == znp.std(samples)

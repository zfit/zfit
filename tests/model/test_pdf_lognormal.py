#  Copyright (c) 2024 zfit
import numpy as np


def test_pdf_lognormal():
    import zfit

    mu = zfit.Parameter("mu", 3.0)
    sigma = zfit.Parameter("sigma", 1.0)
    obs = zfit.Space("obs1", limits=(0, 10))
    lognormal = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)
    probs1 = lognormal.pdf(x=[0.1, 3, 6, 9])
    assert np.all(probs1 >= 0)
    assert np.all(np.isfinite(probs1))
    sample1 = lognormal.sample(1000)
    assert len(sample1.value()) == 1000
    probs2 = lognormal.pdf(x=sample1)
    assert np.all(probs2 >= 0)

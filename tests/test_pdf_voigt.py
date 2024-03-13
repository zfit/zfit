#  Copyright (c) 2023 zfit
import numpy as np
import zfit
from zfit import z
from scipy.special import voigt_profile


def test_voigt1():
    mean1_true = 1.0
    std1_true = 2.0
    width1_true = 1.4

    obs1 = "obs1"
    limits1 = (-10, 10)
    obs1 = zfit.Space(obs1, limits=limits1)

    test_values = z.random.uniform(minval=-10, maxval=10, shape=(1000,))

    mean = zfit.Parameter("mean", mean1_true, -10, 10)
    sigma = zfit.Parameter("sigma", std1_true, 0.1, 10)
    gamma = zfit.Parameter("gamma", width1_true, 0.1, 10)
    voigt1 = zfit.pdf.Voigt(m=mean, sigma=sigma, gamma=gamma, obs=obs1)

    probs1 = voigt1.pdf(x=test_values, norm=False)
    np.testing.assert_array_less(0, probs1)
    sample1 = voigt1.sample(100)
    assert len(sample1.value()) == 100

    probs2 = voigt_profile(test_values - mean1_true, std1_true, width1_true)
    np.testing.assert_allclose(probs1, probs2, rtol=1e-10, atol=1e-08)

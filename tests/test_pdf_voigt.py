#  Copyright (c) 2024 zfit
import numpy as np
import zfit
from zfit import z
import zfit.z.numpy as znp
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

    func = lambda: znp.sum(
        zfit.pdf.Voigt(m=mean, sigma=sigma, gamma=gamma, obs=obs1).pdf(x=test_values)
    )

    num_gradients = z.math.numerical_gradient(func, params=[mean, sigma, gamma])
    tf_gradients = z.math.autodiff_gradient(func, params=[mean, sigma, gamma])
    np.testing.assert_allclose(num_gradients, tf_gradients, rtol=1e-6)

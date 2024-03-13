#  Copyright (c) 2023 zfit

import zfit
from zfit import z

mean1_true = 1.0
std1_true = 1.5
width1_true = 1.4

norm_range1 = (-4.0, 2.0)

obs1 = "obs1"
limits1 = zfit.Space(obs=obs1, limits=(-10, 10))
obs1 = zfit.Space(obs1, limits=limits1)


def test_voigt1():
    test_values = z.random.uniform(minval=-2, maxval=4, shape=(100,))
    mean = zfit.Parameter("mean", mean1_true, -10, 10)
    sigma = zfit.Parameter("sigma", std1_true, 0.1, 10)
    gamma = zfit.Parameter("gamma", width1_true, 0.1, 10)
    voigt1 = zfit.pdf.Voigt(m=mean, sigma=sigma, gamma=gamma, obs=obs1)

    probs1 = voigt1.pdf(x=test_values)
    # TODO: add scipy dist?
    probs1 = probs1.numpy()
    assert all(probs1) > 0
    sample1 = voigt1.sample(100)
    assert len(sample1.numpy()) == 100

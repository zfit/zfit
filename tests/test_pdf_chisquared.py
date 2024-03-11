#  Copyright (c) 2024 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
from zfit.models.dist_tfp import ChiSquared

N_true = 2

obs = zfit.Space(obs="Nobs", limits=(0, 200))

test_values = np.random.uniform(low=0, high=100, size=100)


def test_chisquared():
    N = Parameter("N", N_true)
    chi2 = ChiSquared(obs=obs, ndof=N)

    probs1 = chi2.pdf(x=test_values)
    probs1 = probs1.numpy()

    samples = chi2.sample(10000).numpy()

    assert np.mean(samples) == pytest.approx(N_true, rel=0.05)
    assert np.std(samples) == pytest.approx(np.sqrt(2 * N_true), rel=0.05)

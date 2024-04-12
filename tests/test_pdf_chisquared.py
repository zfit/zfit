#  Copyright (c) 2024 zfit
import pytest

import zfit
import zfit.z.numpy as znp


def test_chisquared():

    N_true = 5
    N = zfit.Parameter("N", N_true)

    obs = zfit.Space(obs="obs", limits=(0, 100))
    chi2 = zfit.pdf.ChiSquared(obs=obs, ndof=N)

    samples = chi2.sample(10000)["obs"]

    assert pytest.approx(N_true, rel=0.05) == znp.mean(samples)
    assert pytest.approx((2 * N_true) ** 0.5, rel=0.05) == znp.std(samples)

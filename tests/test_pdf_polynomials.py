#  Copyright (c) 2019 zfit


import pytest
import zfit
from zfit import ztf

from zfit.core.testing import setup_function, teardown_function, tester
import numpy as np

obs1_random = zfit.Space(obs="obs1", limits=(-0.9, 0.85))
obs1 = zfit.Space(obs="obs1", limits=(-1, 1))


def test_legendre_polynomial():
    legendre = zfit.pdf.Legendre(obs=obs1, coeffs=[0.1, 1.0, 0.6])
    test_size = 100
    probs = legendre.pdf(np.random.uniform(obs1.lower[0], obs1.upper[0], size=test_size))
    probs_np = zfit.run(probs)
    assert probs_np.shape[0] == test_size
    integral = legendre.integrate(limits=obs1, norm_range=False)
    numerical_integral = legendre.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(numerical_integral), rel=1e-2) == zfit.run(integral)
    integral2 = legendre.integrate(limits=obs1_random, norm_range=False)
    numerical_integral2 = legendre.numeric_integrate(limits=obs1_random, norm_range=False)
    assert pytest.approx(zfit.run(numerical_integral2), rel=1e-2) == zfit.run(integral2)


def test_chebyshev():
    chebyshev = zfit.pdf.Chebyshev(obs=obs1, coeffs=[1.1, 1.5, 0.6])

    integral = chebyshev.integrate(limits=obs1, norm_range=False)
    numerical_integral = chebyshev.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=1e-3) == zfit.run(numerical_integral)

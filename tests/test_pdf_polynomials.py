#  Copyright (c) 2019 zfit
import copy

import pytest
import zfit
from zfit import ztf

from zfit.core.testing import setup_function, teardown_function, tester
import numpy as np

obs1_random = zfit.Space(obs="obs1", limits=(-0.9, 0.85))
obs1 = zfit.Space(obs="obs1", limits=(-1, 1))

coeffs_parametrization = [
    1.4,
    [0.6],
    [1.42, 1.9],
    [0.1, 2.0, 0.6],
    [21.1, 11.4, 3.6, 4.1],
    [1.1, 1.42, 1.6, 0.1, 0.7],
    [11.1, 1.4, 5.6, 3.1, 18.1, 3.1],
]

rel_integral = 3e-2


@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_legendre_polynomial(coeffs):
    coeffs = copy.copy(coeffs)  # TODO(Mayou36): check why neeed? reference kept and object muted? Where?

    legendre = zfit.pdf.Legendre(obs=obs1, coeffs=coeffs)
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


@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_chebyshev(coeffs):
    coeffs = copy.copy(coeffs)
    chebyshev = zfit.pdf.Chebyshev(obs=obs1, coeffs=coeffs)

    integral = chebyshev.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = chebyshev.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=rel_integral) == zfit.run(numerical_integral)


@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_chebyshev2(coeffs):
    coeffs = copy.copy(coeffs)

    chebyshev2 = zfit.pdf.Chebyshev2(obs=obs1, coeffs=coeffs)

    integral = chebyshev2.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = chebyshev2.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=rel_integral) == zfit.run(numerical_integral)


@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_laguerre(coeffs):
    coeffs = copy.copy(coeffs)

    laguerre = zfit.pdf.Laguerre(obs=obs1, coeffs=coeffs)

    integral = laguerre.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = laguerre.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=rel_integral) == zfit.run(numerical_integral)


@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_hermite(coeffs):
    coeffs = copy.copy(coeffs)

    hermite = zfit.pdf.Laguerre(obs=obs1, coeffs=coeffs)

    integral = hermite.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = hermite.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=rel_integral) == zfit.run(numerical_integral)

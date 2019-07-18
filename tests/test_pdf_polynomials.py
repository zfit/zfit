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

rel_integral = 6e-2

poly_pdfs = [zfit.pdf.Legendre,
             zfit.pdf.Chebyshev,
             zfit.pdf.Chebyshev2,
             zfit.pdf.Hermite,
             zfit.pdf.Laguerre]


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


@pytest.mark.parametrize("poly_pdf", poly_pdfs)
@pytest.mark.parametrize("coeffs", coeffs_parametrization)
def test_polynomials(poly_pdf, coeffs):
    coeffs = copy.copy(coeffs)
    polynomial = poly_pdf(obs=obs1, coeffs=coeffs)

    integral = polynomial.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm_range=False)
    analytic_integral = zfit.run(integral)
    assert pytest.approx(analytic_integral, rel=rel_integral) == zfit.run(numerical_integral)

    # test_integral =

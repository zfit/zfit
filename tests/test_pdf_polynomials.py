#  Copyright (c) 2019 zfit
import copy

import pytest
import zfit
from zfit import ztf
import tensorflow as tf

from zfit.core.testing import setup_function, teardown_function, tester

import numpy as np

# obs1_random = zfit.Space(obs="obs1", limits=(-1.05, 1.05))
obs1_random = zfit.Space(obs="obs1", limits=(-1.5, 1.2))
obs1 = zfit.Space(obs="obs1", limits=(-1, 1))

coeffs_parametrization = [
    1.4,
    [0.6],
    [1.42, 1.2],
    [0.2, 0.8, 0.5],
    [8.1, 1.4, 3.6, 4.1],
    [1.1, 1.42, 1.2, 0.4, 0.7],
    [11.1, 1.4, 5.6, 3.1, 18.1, 3.1],
]

rel_integral = 6e-2
default_sampling = 60000

poly_pdfs = [(zfit.pdf.Legendre, default_sampling),
             (zfit.pdf.Chebyshev, default_sampling),
             (zfit.pdf.Chebyshev2, default_sampling),
             (zfit.pdf.Hermite, default_sampling * 20),
             (zfit.pdf.Laguerre, default_sampling * 20)]


@pytest.mark.parametrize("poly_cfg", poly_pdfs)
@pytest.mark.parametrize("coeffs", coeffs_parametrization)
@pytest.mark.flaky(3)
def test_polynomials(poly_cfg, coeffs):
    coeffs = copy.copy(coeffs)
    poly_pdf, n_sampling = poly_cfg
    polynomial = poly_pdf(obs=obs1, coeffs=coeffs)
    polynomial2 = poly_pdf(obs=obs1, coeffs=coeffs)

    polynomial_coeff0 = poly_pdf(obs=obs1, coeffs=coeffs, coeff0=1.)
    lower, upper = obs1.limit1d
    x = np.random.uniform(size=(1000,), low=lower, high=upper)
    y_poly = polynomial.pdf(x)
    y_poly_u = polynomial.unnormalized_pdf(x)
    y_poly2 = polynomial2.pdf(x)
    y_poly2_u = polynomial2.unnormalized_pdf(x)
    y_poly_coeff0 = polynomial_coeff0.pdf(x)
    y_poly_coeff0_u = polynomial_coeff0.unnormalized_pdf(x)
    y_poly_np, y_poly2_np, y_poly_coeff0_np = zfit.run([y_poly, y_poly2, y_poly_coeff0])
    y_polyu_np, y_poly2u_np, y_polyu_coeff0_np = zfit.run([y_poly_u, y_poly2_u, y_poly_coeff0_u])
    np.testing.assert_allclose(y_polyu_np, y_poly2u_np)
    np.testing.assert_allclose(y_polyu_np, y_polyu_coeff0_np)
    np.testing.assert_allclose(y_poly_np, y_poly2_np)
    np.testing.assert_allclose(y_poly_np, y_poly_coeff0_np)

    # test 1 to 1 range
    integral = polynomial.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm_range=False)
    analytic_integral = zfit.run(integral)
    assert pytest.approx(analytic_integral, rel=rel_integral) == zfit.run(numerical_integral)

    # test with different range scaling
    polynomial = poly_pdf(obs=obs1_random, coeffs=coeffs)

    # test with limits != space
    integral = polynomial.analytic_integrate(limits=obs1, norm_range=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm_range=False)
    analytic_integral = zfit.run(integral)
    assert pytest.approx(analytic_integral, rel=rel_integral) == zfit.run(numerical_integral)

    # test with limits == space
    integral = polynomial.analytic_integrate(limits=obs1_random, norm_range=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1_random, norm_range=False)
    analytic_integral = zfit.run(integral)
    assert pytest.approx(analytic_integral, rel=rel_integral) == zfit.run(numerical_integral)

    lower, upper = obs1_random.limit1d
    test_integral = np.average(zfit.run(polynomial.unnormalized_pdf(tf.random.uniform((n_sampling,), lower, upper)))) \
                    * obs1_random.area()
    assert pytest.approx(analytic_integral, rel=rel_integral * 3) == test_integral

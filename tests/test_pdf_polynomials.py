#  Copyright (c) 2023 zfit
import copy

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import z

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

rel_integral = 7e-2
default_sampling = 100000

poly_pdfs = [
    (zfit.pdf.Legendre, default_sampling),
    (zfit.pdf.Chebyshev, default_sampling),
    (zfit.pdf.Chebyshev2, default_sampling),
    (zfit.pdf.Hermite, default_sampling * 20),
    (zfit.pdf.Laguerre, default_sampling * 20),
]


@pytest.mark.parametrize("poly_cfg", poly_pdfs)
@pytest.mark.parametrize("coeffs", coeffs_parametrization)
@pytest.mark.flaky(3)
def test_polynomials(poly_cfg, coeffs):
    coeffs = copy.copy(coeffs)
    poly_pdf, n_sampling = poly_cfg
    polynomial = poly_pdf(obs=obs1, coeffs=coeffs)
    polynomial2 = poly_pdf(obs=obs1, coeffs=coeffs)

    polynomial_coeff0 = poly_pdf(obs=obs1, coeffs=coeffs, coeff0=1.0)
    lower, upper = obs1.rect_limits
    x = np.random.uniform(size=(1000,), low=lower[0], high=upper[0])
    y_poly = polynomial.pdf(x)
    y_poly_u = polynomial.pdf(x, norm=False)
    y_poly2 = polynomial2.pdf(x)
    y_poly2_u = polynomial2.pdf(x, norm=False)
    y_poly_coeff0 = polynomial_coeff0.pdf(x)
    y_poly_coeff0_u = polynomial_coeff0.pdf(x, norm=False)
    y_poly_np, y_poly2_np, y_poly_coeff0_np = [
        y_poly.numpy(),
        y_poly2.numpy(),
        y_poly_coeff0.numpy(),
    ]
    y_polyu_np, y_poly2u_np, y_polyu_coeff0_np = [
        y_poly_u.numpy(),
        y_poly2_u.numpy(),
        y_poly_coeff0_u.numpy(),
    ]
    np.testing.assert_allclose(y_polyu_np, y_poly2u_np)
    np.testing.assert_allclose(y_polyu_np, y_polyu_coeff0_np)
    np.testing.assert_allclose(y_poly_np, y_poly2_np)
    np.testing.assert_allclose(y_poly_np, y_poly_coeff0_np)

    # test 1 to 1 range
    integral = polynomial.analytic_integrate(limits=obs1, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm=False)
    analytic_integral = integral.numpy()
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral.numpy()
    )

    # test with different range scaling
    polynomial = poly_pdf(obs=obs1_random, coeffs=coeffs)

    # test with limits != space
    integral = polynomial.analytic_integrate(limits=obs1, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm=False)
    analytic_integral = integral.numpy()
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral.numpy()
    )

    # test with limits == space
    integral = polynomial.analytic_integrate(limits=obs1_random, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1_random, norm=False)
    analytic_integral = integral.numpy()
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral.numpy()
    )

    lower, upper = obs1_random.limit1d
    sample = z.random.uniform((n_sampling, 1), lower, upper, dtype=tf.float64)
    test_integral = (
        np.average(polynomial.pdf(sample, norm=False)) * obs1_random.rect_area()
    )
    assert pytest.approx(analytic_integral, rel=rel_integral * 3) == test_integral

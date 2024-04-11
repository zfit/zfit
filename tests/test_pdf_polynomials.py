#  Copyright (c) 2024 zfit
import copy

import numpy as np
import pytest
import tensorflow as tf
from numba_stats import bernstein as bernstein_numba

import zfit
from zfit import z
import zfit.z.numpy as znp

obs1_random = zfit.Space(obs="obs1", limits=(-1.5, 1.2))
obs1 = zfit.Space(obs="obs1", limits=(-1, 1))
obs2_random = zfit.Space(obs="obs2", limits=(0.5, 1.8))
obs2 = zfit.Space(obs="obs2", limits=(0, 1))

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
    (zfit.pdf.Bernstein, default_sampling),
]


@pytest.mark.parametrize("poly_cfg", poly_pdfs)
@pytest.mark.parametrize("coeffs", coeffs_parametrization)
@pytest.mark.flaky(3)
def test_polynomials(poly_cfg, coeffs):
    coeffs = copy.copy(coeffs)
    poly_pdf, n_sampling = poly_cfg
    polynomial = poly_pdf(obs=obs1, coeffs=coeffs)
    polynomial2 = poly_pdf(obs=obs1, coeffs=coeffs)

    if poly_pdf == zfit.pdf.Bernstein:
        polynomial_coeff0 = poly_pdf(obs=obs1, coeffs=coeffs)
    else:
        polynomial_coeff0 = poly_pdf(obs=obs1, coeffs=coeffs, coeff0=1.0)
    lower, upper = obs1.v1.limits
    x = np.random.uniform(size=(1000,), low=lower, high=upper)
    y_poly = polynomial.pdf(x)
    y_poly_u = polynomial.pdf(x, norm=False)
    y_poly2 = polynomial2.pdf(x)
    y_poly2_u = polynomial2.pdf(x, norm=False)
    y_poly_coeff0 = polynomial_coeff0.pdf(x)
    y_poly_coeff0_u = polynomial_coeff0.pdf(x, norm=False)
    y_poly_np, y_poly2_np, y_poly_coeff0_np = [
        y_poly,
        y_poly2,
        y_poly_coeff0,
    ]
    y_polyu_np, y_poly2u_np, y_polyu_coeff0_np = [
        y_poly_u,
        y_poly2_u,
        y_poly_coeff0_u,
    ]
    np.testing.assert_allclose(y_polyu_np, y_poly2u_np)
    np.testing.assert_allclose(y_polyu_np, y_polyu_coeff0_np)
    np.testing.assert_allclose(y_poly_np, y_poly2_np)
    np.testing.assert_allclose(y_poly_np, y_poly_coeff0_np)

    # test 1 to 1 range
    polynomial = poly_pdf(obs=obs1, coeffs=coeffs)
    integral = polynomial.analytic_integrate(limits=obs1, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    # test 0 to 1 range
    polynomial = poly_pdf(obs=obs2, coeffs=coeffs)
    integral = polynomial.analytic_integrate(limits=obs2, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs2, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    # test with different range scaling, 1 to 1
    polynomial = poly_pdf(obs=obs1_random, coeffs=coeffs)

    # test with limits != space
    integral = polynomial.analytic_integrate(limits=obs1, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    # test with limits == space
    integral = polynomial.analytic_integrate(limits=obs1_random, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs1_random, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    lower, upper = obs1_random.limit1d
    sample = z.random.uniform((n_sampling, 1), lower, upper, dtype=tf.float64)
    test_integral = (
        np.average(polynomial.pdf(sample, norm=False)) * obs1_random.volume
    )
    assert pytest.approx(analytic_integral, rel=rel_integral * 3) == test_integral

    # test with different range scaling, 0 to 1
    polynomial = poly_pdf(obs=obs2_random, coeffs=coeffs)

    # test with limits != space
    integral = polynomial.analytic_integrate(limits=obs2, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs2, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    # test with limits == space
    integral = polynomial.analytic_integrate(limits=obs2_random, norm=False)
    numerical_integral = polynomial.numeric_integrate(limits=obs2_random, norm=False)
    analytic_integral = integral
    assert (
        pytest.approx(analytic_integral, rel=rel_integral) == numerical_integral
    )

    lower, upper = obs2_random.limit1d
    sample = z.random.uniform((n_sampling, 1), lower, upper, dtype=tf.float64)
    test_integral = (
        np.average(polynomial.pdf(sample, norm=False)) * obs2_random.volume
     )
    assert pytest.approx(analytic_integral, rel=rel_integral * 3) == test_integral




@pytest.mark.parametrize("coeffs", coeffs_parametrization)
@pytest.mark.parametrize("obs", [obs1, obs2, obs1_random, obs2_random])
def test_bernstein(coeffs, obs):
    zfit.run.set_graph_mode(False)
    bernstein = zfit.pdf.Bernstein(obs=obs, coeffs=coeffs)
    lower, upper = obs.limit1d

    assert pytest.approx(
        np.atleast_1d(bernstein_numba.density(0.8, beta=coeffs, xmin=lower, xmax=upper)), rel=1e-5
    ) == bernstein.pdf(0.8, norm=False)
    test_values = znp.linspace(lower, upper, 10_000)
    np.testing.assert_allclose(
        bernstein.pdf(test_values, norm=False),
        bernstein_numba.density(test_values, beta=coeffs, xmin=lower, xmax=upper),
        rtol=1e-5,
    )

    sample = bernstein.sample(1000)
    assert all(np.isfinite(sample.value())), "Some samples from the Bernstein PDF are NaN or infinite"
    assert sample.n_events == 1000
    assert all(tf.logical_and(lower <= sample.value(), sample.value() <= upper))


    full_interval_analytic = bernstein.analytic_integrate(obs, norm=False)
    full_interval_numeric = bernstein.numeric_integrate(obs, norm=False)
    numba_stats_full_integral = bernstein_numba.integral(x=upper, beta=coeffs, xmin=lower, xmax=upper) - bernstein_numba.integral(
        x=lower, beta=coeffs, xmin=lower, xmax=upper
    )
    assert pytest.approx(full_interval_numeric, 1e-4) == full_interval_analytic
    assert pytest.approx(numba_stats_full_integral, 1e-6) == full_interval_analytic
    assert pytest.approx(numba_stats_full_integral, 1e-6) == full_interval_numeric

    analytic_integral = bernstein.analytic_integrate(limits=(0.6, 0.9), norm=False)
    numeric_integral = bernstein.numeric_integrate(limits=(0.6, 0.9), norm=False)
    numba_stats_integral = bernstein_numba.integral(x=0.9, beta=coeffs, xmin=lower, xmax=upper) - bernstein_numba.integral(
        x=0.6, beta=coeffs, xmin=lower, xmax=upper
    )
    assert pytest.approx(numeric_integral, 1e-5) == analytic_integral
    assert pytest.approx(numba_stats_integral, 1e-5) == analytic_integral
    assert pytest.approx(numba_stats_integral, 1e-5) == numeric_integral

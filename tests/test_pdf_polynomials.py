#  Copyright (c) 2019 zfit


import pytest
import zfit
from zfit import ztf

from zfit.core.testing import setup_function, teardown_function, tester

obs1 = zfit.Space(obs="obs1", limits=(-0.9, 0.85))


def test_legendre_polynomial():
    legendre = zfit.pdf.Legendre(obs=obs1, coeffs=[0.1, 1.5, 0.6])

    integral = legendre.integrate(limits=obs1, norm_range=False)
    numerical_integral = legendre.numeric_integrate(limits=obs1, norm_range=False)
    assert zfit.run(numerical_integral) == pytest.approx(zfit.run(integral))


def test_legendre_chebyshev():
    chebyshev = zfit.pdf.Chebyshev(obs=obs1, coeffs=[0.1, 0.5, 0.6, 0.3])

    integral = chebyshev.integrate(limits=obs1, norm_range=False)
    numerical_integral = chebyshev.numeric_integrate(limits=obs1, norm_range=False)
    assert pytest.approx(zfit.run(integral), rel=1e-3) == zfit.run(numerical_integral)

#  Copyright (c) 2024 zfit
import numpy as np
import pytest
from scipy import integrate

import zfit
from zfit.core.testing import tester
from zfit.models.physics import GaussExpTail, GeneralizedGaussExpTail

mu = -0.3
sigma = 1.1

sigmal = 1.1
sigmar = 1.1

alphal = 0.8
nl = alphal

alphar = 1.4
nr = alphar

bounds = (-6, 4)
lbounds = (bounds[0], mu)
rbounds = (mu, bounds[1])

def _numpy_gaussexptail_pdf(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * np.sign(alpha)
    abs_alpha = np.abs(alpha)
    if t > -abs_alpha:
        return np.exp(-0.5 * t ** 2)
    else:
        return np.exp(-0.5 * abs_alpha ** 2) * np.exp(n * (t + abs_alpha))

numpy_gaussexptail_pdf = np.vectorize(_numpy_gaussexptail_pdf, excluded=["mu", "sigma", "alpha", "n"])

def _numpy_generalizedgaussexptail_pdf(x, mu, sigmal, alphal, nl, sigmar, alphar, nr):
    if x < mu:
        return _numpy_gaussexptail_pdf(x, mu, sigmal, alphal, nl)
    else:
        return _numpy_gaussexptail_pdf(x, mu, sigmar, -alphar, nr)

numpy_generalizedgaussexptail_pdf = np.vectorize(_numpy_generalizedgaussexptail_pdf, excluded=["mu", "sigmal", "alphal", "nl", "sigmar", "alphar", "nr"])


def _gaussexptail_params_factory(name_add=""):
    mu_ = zfit.Parameter(f"mu_gaussexptail{name_add}", mu)
    sigma_ = zfit.Parameter(f"sigma_gaussexptail{name_add}", sigma)
    alphal_ = zfit.Parameter(f"alphal_gaussexptail{name_add}", alphal)
    return {"mu": mu_, "sigma": sigma_, "alpha": alphal_}

tester.register_pdf(pdf_class=GaussExpTail, params_factories=_gaussexptail_params_factory)



def sample_testing(pdf):
    sample = pdf.sample(n=1000, limits=(-0.5, 1.5)).value()
    assert not any(np.isnan(sample))


def eval_testing(pdf, x):
    probs = pdf.pdf(x, norm=False)
    assert probs.shape.rank == 1
    assert probs.shape[0] == x.shape[0]
    assert not np.any(np.isnan(probs))
    return probs


def test_gaussexptail_integral():
    obs = zfit.Space("x", limits=bounds)

    mu_ = zfit.Parameter("mu", mu)
    sigma_ = zfit.Parameter("sigma", sigma)
    alphal_ = zfit.Parameter("alphal", alphal)

    gaussexptail = GaussExpTail(obs=obs, mu=mu_, sigma=sigma_, alpha=alphal_)
    int_limits = (-1, 3)
    integral_numeric = gaussexptail.numeric_integrate(limits=int_limits, norm=False)
    integral = gaussexptail.analytic_integrate(limits=int_limits, norm=False)
    integral_scipy = integrate.quad(numpy_gaussexptail_pdf, *int_limits, args=(mu, sigma, alphal, nl))[0]

    assert pytest.approx(integral_numeric, 1e-5) == integral
    assert pytest.approx(integral_scipy, 1e-5) == integral

    rnd_limits = sorted(list(np.random.uniform(*bounds, 13)) + list(bounds))
    integrals = [
        gaussexptail.integrate((low, up), norm=False)
        for low, up in zip(rnd_limits[:-1], rnd_limits[1:])
    ]

    integral = np.sum(integrals)
    integral_full = gaussexptail.integrate(bounds, norm=False)
    assert pytest.approx(integral_full) == integral


def test_gaussexptail_generalizedgaussexptail():
    obs = zfit.Space("x", limits=bounds)

    mu_ = zfit.Parameter("mu", mu)
    sigma_ = zfit.Parameter("sigma", sigma)
    sigmal_ = zfit.Parameter("sigmal", sigmal)
    alphal_ = zfit.Parameter("alphal", alphal)
    sigmar_ = zfit.Parameter("sigmar", sigmar)
    alphar_ = zfit.Parameter("alphar", alphar)

    gaussexptaill = GaussExpTail(obs=obs, mu=mu_, sigma=sigmal_, alpha=alphal_)
    gaussexptailr = GaussExpTail(obs=obs, mu=mu_, sigma=sigmar_, alpha=-alphar_)
    generalizedgaussexptail = GeneralizedGaussExpTail(
        obs=obs,
        mu=mu_,
        sigmal=sigmal_,
        alphal=alphal_,
        sigmar=sigmar_,
        alphar=alphar_,
    )

    sample_testing(gaussexptaill)
    sample_testing(gaussexptailr)
    sample_testing(generalizedgaussexptail)

    x = np.random.normal(mu, sigma, size=10_000)

    probsl = eval_testing(gaussexptaill, x)
    probsr = eval_testing(gaussexptailr, x)

    assert not any(np.isnan(probsl))
    assert not any(np.isnan(probsr))

    probsl_numpy = numpy_gaussexptail_pdf(x, mu, sigmal, alphal, nl)
    probsr_numpy = numpy_gaussexptail_pdf(x, mu, sigmar, -alphar, nr)

    ratio_l = probsl_numpy / probsl
    ratio_r = probsr_numpy / probsr

    np.testing.assert_allclose(ratio_l, 1.0, rtol=5e-7)
    np.testing.assert_allclose(ratio_r, 1.0, rtol=5e-7)

    kwargs = dict(limits=(-5.0, mu), norm=lbounds)
    intl = gaussexptaill.integrate(**kwargs) - generalizedgaussexptail.integrate(**kwargs)
    assert pytest.approx(intl, abs=1e-3) == 0.0
    intl = gaussexptailr.integrate(**kwargs) - generalizedgaussexptail.integrate(**kwargs)
    assert pytest.approx(intl, abs=2e-4) != 0.0

    kwargs = dict(limits=(mu, 2.0), norm=rbounds)
    intr = gaussexptailr.integrate(**kwargs) - generalizedgaussexptail.integrate(**kwargs)
    assert pytest.approx(intr, abs=1e-3) == 0.0
    intr = gaussexptaill.integrate(**kwargs) - generalizedgaussexptail.integrate(**kwargs)
    assert pytest.approx(intr, abs=1e-3) != 0.0

    integrall = gaussexptaill.integrate(limits=bounds, norm=False)
    integralr = gaussexptailr.integrate(limits=bounds, norm=False)
    integral = generalizedgaussexptail.integrate(limits=bounds, norm=False)
    integrall_scipy = integrate.quad(numpy_gaussexptail_pdf, *bounds, args=(mu, sigmal, alphal, nl))[0]
    integralr_scipy = integrate.quad(numpy_gaussexptail_pdf, *bounds, args=(mu, sigmar, -alphar, nr))[0]
    integral_scipy = integrate.quad(numpy_generalizedgaussexptail_pdf, *bounds, args=(mu, sigmal, alphal, nl, sigmar, alphar, nr))[0]

    np.testing.assert_allclose(integrall, integrall_scipy)
    np.testing.assert_allclose(integralr, integralr_scipy)
    np.testing.assert_allclose(integral, integral_scipy)

    xl = x[x <= mu]
    xr = x[x > mu]

    probs_generalizedgaussexptail_l = eval_testing(generalizedgaussexptail, xl)
    probs_generalizedgaussexptail_r = eval_testing(generalizedgaussexptail, xr)

    probsl_numpy = numpy_gaussexptail_pdf(xl, mu, sigmal, alphal, nl)
    probsr_numpy = numpy_gaussexptail_pdf(xr, mu, sigmar, -alphar, nr)

    ratio_l = probsl_numpy / probs_generalizedgaussexptail_l
    ratio_r = probsr_numpy / probs_generalizedgaussexptail_r

    np.testing.assert_allclose(ratio_l, 1.0, rtol=5e-7)
    np.testing.assert_allclose(ratio_r, 1.0, rtol=5e-7)

    rnd_limits = sorted(list(np.random.uniform(*bounds, 130)) + list(bounds))
    integrals = []
    for low, up in zip(rnd_limits[:-1], rnd_limits[1:]):
        integrals.append(generalizedgaussexptail.integrate((low, up), norm=False))

    integral = np.sum(integrals)
    integral_full = generalizedgaussexptail.integrate(bounds, norm=False)
    assert pytest.approx(integral_full) == integral

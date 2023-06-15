#  Copyright (c) 2023 zfit
import numpy as np
import pytest
from scipy.stats import crystalball

import zfit
from zfit.core.testing import tester
from zfit.models.physics import CrystalBall, DoubleCB

mu = -0.3
sigma = 1.1

alphal = 0.8
nl = 2.0

alphar = 1.4
nr = 4.0

bounds = (-6, 4)
lbounds = (bounds[0], mu)
rbounds = (mu, bounds[1])


def _cb_params_factory(name_add=""):
    mu_ = zfit.Parameter(f"mu_cb{name_add}", mu)
    sigma_ = zfit.Parameter(f"sigma_cb{name_add}", sigma)
    alphal_ = zfit.Parameter(f"alphal_cb{name_add}", alphal)
    nl_ = zfit.Parameter(f"nl_cb{name_add}", nl)
    return {"mu": mu_, "sigma": sigma_, "alpha": alphal_, "n": nl_}


tester.register_pdf(pdf_class=CrystalBall, params_factories=_cb_params_factory)


def sample_testing(pdf):
    sample = pdf.sample(n=1000, limits=(-0.5, 1.5))
    sample_np = sample.numpy()
    assert not any(np.isnan(sample_np))


def eval_testing(pdf, x):
    probs = pdf.pdf(x)
    assert probs.shape.rank == 1
    assert probs.shape[0] == x.shape[0]
    probs = zfit.run(probs)
    assert not np.any(np.isnan(probs))
    return probs


def test_cb_integral():
    obs = zfit.Space("x", limits=bounds)

    mu_ = zfit.Parameter("mu", mu)
    sigma_ = zfit.Parameter("sigma", sigma)
    alphal_ = zfit.Parameter("alphal", alphal)
    nl_ = zfit.Parameter("nl", nl)

    cbl = CrystalBall(obs=obs, mu=mu_, sigma=sigma_, alpha=alphal_, n=nl_)
    int_limits = (-1, 3)
    integral_numeric = cbl.numeric_integrate(limits=int_limits, norm=False)

    integral = cbl.analytic_integrate(limits=int_limits, norm=False)
    integral_numeric = zfit.run(integral_numeric)
    integral = zfit.run(integral)

    assert pytest.approx(integral_numeric, 1e-5) == integral

    rnd_limits = sorted(list(np.random.uniform(*bounds, 13)) + list(bounds))
    integrals = [
        cbl.integrate((low, up), norm=False)
        for low, up in zip(rnd_limits[:-1], rnd_limits[1:])
    ]

    integral = np.sum(integrals)
    integral_full = zfit.run(cbl.integrate(bounds, norm=False))
    assert pytest.approx(float(integral_full)) == float(integral)


def test_cb_dcb():
    obs = zfit.Space("x", limits=bounds)

    mu_ = zfit.Parameter("mu", mu)
    sigma_ = zfit.Parameter("sigma", sigma)
    alphal_ = zfit.Parameter("alphal", alphal)
    nl_ = zfit.Parameter("nl", nl)
    alphar_ = zfit.Parameter("alphar", alphar)
    nr_ = zfit.Parameter("nr", nr)

    cbl = CrystalBall(obs=obs, mu=mu_, sigma=sigma_, alpha=alphal_, n=nl_)
    cbr = CrystalBall(obs=obs, mu=mu_, sigma=sigma_, alpha=-alphar_, n=nr_)
    dcb = DoubleCB(
        obs=obs, mu=mu_, sigma=sigma_, alphal=alphal_, nl=nl_, alphar=alphar_, nr=nr_
    )

    sample_testing(cbl)
    sample_testing(cbr)
    sample_testing(dcb)

    x = np.random.normal(mu, sigma, size=10000)

    probsl = eval_testing(cbl, x)
    probsr = eval_testing(cbr, x)

    assert not any(np.isnan(probsl))
    assert not any(np.isnan(probsr))

    probsl_scipy = crystalball.pdf(x, beta=alphal, m=nl, loc=mu, scale=sigma)
    probsr_scipy = crystalball.pdf(-x + 2 * mu, beta=alphar, m=nr, loc=mu, scale=sigma)

    # We take the ration as the normalization is not fixed
    ratio_l = probsl_scipy / probsl
    ratio_r = probsr_scipy / probsr

    assert np.allclose(ratio_l, ratio_l[0])
    assert np.allclose(ratio_r, ratio_r[0])

    kwargs = dict(limits=(-5.0, mu), norm_range=lbounds)
    intl = cbl.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(intl.numpy(), abs=1e-3) == 0.0
    intl = cbr.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(intl.numpy(), abs=1e-3) != 0

    # TODO: update test to fixed DCB integral
    kwargs = dict(limits=(mu, 2.0), norm_range=rbounds)
    dcb_integr1 = dcb.integrate(**kwargs)
    intr = cbr.integrate(**kwargs) - dcb_integr1
    assert pytest.approx(intr.numpy(), abs=1e-3) == 0.0
    intr = cbl.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(intr.numpy(), abs=1e-3) != 0.0

    xl = x[x <= mu]
    xr = x[x > mu]

    probs_dcb_l = eval_testing(dcb, xl)
    probs_dcb_r = eval_testing(dcb, xr)

    probsl_scipy = crystalball.pdf(xl, beta=alphal, m=nl, loc=mu, scale=sigma)
    probsr_scipy = crystalball.pdf(-xr + 2 * mu, beta=alphar, m=nr, loc=mu, scale=sigma)

    ratio_l = probsl_scipy / probs_dcb_l
    ratio_r = probsr_scipy / probs_dcb_r

    assert np.allclose(ratio_l, ratio_l[0])
    assert np.allclose(ratio_r, ratio_r[0])

    rnd_limits = sorted(list(np.random.uniform(*bounds, 130)) + list(bounds))
    integrals = []
    for low, up in zip(rnd_limits[:-1], rnd_limits[1:]):
        integrals.append(dcb.integrate((low, up), norm=False))

    integral = np.sum(integrals)
    integral_full = zfit.run(dcb.integrate((bounds[0], up), norm=False))
    assert pytest.approx(float(integral_full)) == float(integral)

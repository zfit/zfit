#  Copyright (c) 2019 zfit

import pytest
import numpy as np
from zfit.models.physics import CrystalBall, DoubleCB
import zfit
from scipy.stats import crystalball
from zfit.core.testing import setup_function, teardown_function, tester

mu = 0.0
sigma = 0.5

alphal = 1.0
nl = 2.0

alphar = 3.0
nr = 4.0

bounds = (-10, 3.0)
lbounds = (bounds[0], mu)
rbounds = (mu, bounds[1])


def _cb_params_factory():
    mu_ = zfit.Parameter('mu_cb', mu)
    sigma_ = zfit.Parameter('sigma_cb', sigma)
    alphal_ = zfit.Parameter('alphal_cb', alphal)
    nl_ = zfit.Parameter('nl_cb', nl)
    return {"mu": mu_, "sigma": sigma_, "alpha": alphal_, "n": nl_}


tester.register_pdf(pdf_class=CrystalBall, params_factories=_cb_params_factory())


def sample_testing(pdf):
    sample = pdf.sample(n=1000, limits=(-0.5, 1.5))
    sample_np = zfit.run(sample)
    assert not any(np.isnan(sample_np))


def eval_testing(pdf, x):
    probs = zfit.run(pdf.pdf(x))
    assert not any(np.isnan(probs))
    return probs


def test_cb_dcb():
    obs = zfit.Space('x', limits=bounds)

    mu_ = zfit.Parameter('mu_cb', mu)
    sigma_ = zfit.Parameter('sigma_cb', sigma)
    alphal_ = zfit.Parameter('alphal_cb', alphal)
    nl_ = zfit.Parameter('nl_cb', nl)
    alphar_ = zfit.Parameter('alphar_cb', alphar)
    nr_ = zfit.Parameter('nr_cb', nr)

    cbl = CrystalBall(obs=obs, mu=mu_, sigma=sigma_, alpha=alphal_, n=nl_)
    cbr = CrystalBall(obs=obs, mu=mu_, sigma=sigma_, alpha=-alphar_, n=nr_)
    dcb = DoubleCB(obs=obs, mu=mu_, sigma=sigma_, alphal=alphal_, nl=nl_, alphar=alphar_, nr=nr_)

    sample_testing(cbl)
    sample_testing(cbr)
    sample_testing(dcb)

    x = np.random.normal(mu, sigma, size=10000)

    probsl = eval_testing(cbl, x)
    probsr = eval_testing(cbr, x)

    assert not any(np.isnan(probsl))
    assert not any(np.isnan(probsr))

    probsl_scipy = crystalball.pdf(x, beta=alphal, m=nl, loc=mu, scale=sigma)
    probsr_scipy = crystalball.pdf(-x, beta=alphar, m=nr, loc=mu, scale=sigma)

    ratio_l = probsl_scipy / probsl
    ratio_r = probsr_scipy / probsr

    assert np.allclose(ratio_l, ratio_l[0])
    assert np.allclose(ratio_r, ratio_r[0])

    kwargs = dict(limits=(-5.0, mu), norm_range=lbounds)
    intl = cbl.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(zfit.run(intl)) == 0.
    intl = cbr.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(zfit.run(intl)) != 0

    kwargs = dict(limits=(mu, 2.0), norm_range=rbounds)
    intr = cbr.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(zfit.run(intr)) == 0.
    intr = cbl.integrate(**kwargs) - dcb.integrate(**kwargs)
    assert pytest.approx(zfit.run(intr)) != 0.

    xl = x[x <= mu]
    xr = x[x > mu]

    probs_dcb_l = eval_testing(dcb, xl)
    probs_dcb_r = eval_testing(dcb, xr)

    probsl_scipy = crystalball.pdf(xl, beta=alphal, m=nl, loc=mu, scale=sigma)
    probsr_scipy = crystalball.pdf(-xr, beta=alphar, m=nr, loc=mu, scale=sigma)

    ratio_l = probsl_scipy / probs_dcb_l
    ratio_r = probsr_scipy / probs_dcb_r

    assert np.allclose(ratio_l, ratio_l[0])
    assert np.allclose(ratio_r, ratio_r[0])

#  Copyright (c) 2024 zfit

import numpy as np
import pytest

import zfit
import zfit.z.numpy as znp
import tensorflow as tf

from zfit import supports, z
from zfit.models.cache import CachedPDF
from zfit.util import ztyping
from zfit.util.exception import AnalyticGradientNotAvailable


class TestPDF(zfit.pdf.BaseFunctor):
    def __init__(self, obs: ztyping.ObsTypeInput, mu, sigma):
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        super().__init__(pdfs=gauss, obs=obs)
        self.pdf_call_counter = tf.Variable(0)
        self.integrate_call_counter = tf.Variable(0)

    @supports(norm="space")
    def _pdf(self, x, norm):
        self.pdf_call_counter.assign_add(1)
        return self.pdfs[0].pdf(x)

    @supports(norm="space")
    def _integrate(self, limits, norm, options=None):
        self.integrate_call_counter.assign_add(1)
        return self.pdfs[0].integrate(limits)


def test_cached_pdf_equals_pdf_without_cache():

    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CachedPDF(gauss)
    x = znp.linspace(-5, 5, 500)
    xhashed = zfit.data.Data.from_numpy(obs=obs, array=x, use_hash=True)
    assert xhashed.hashint is not None
    xunhashed = zfit.data.Data.from_numpy(obs=obs, array=x, use_hash=False)
    assert xunhashed.hashint is None
    assert tf.math.reduce_all(tf.equal(gauss.pdf(xhashed), cached_gauss.pdf(xhashed)))
    # hash not yet used inside cached pdf
    # with pytest.raises(ValueError):
    #     _ = gauss.pdf(xunhashed)


def test_pdf_cache_is_used():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    x = znp.linspace(-5, 5, 500)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(0))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(1))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(1))


def test_pdf_cache_revaluation_if_mu_was_changed():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    x = znp.linspace(-5, 5, 500)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(0))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(1))
    mu.set_value(2.0)
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))


def test_pdf_cache_revaluation_if_sigma_was_changed():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    x = znp.linspace(-5, 5, 500)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(0))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(1))
    sigma.set_value(3.0)
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))


def test_pdf_cache_revaluation_if_x_was_changed():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    x = znp.linspace(-5, 5, 500)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(0))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(1))
    x = znp.linspace(-5, 0, 500)
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))
    cached_test_pdf.pdf(x)
    assert tf.equal(test_pdf.pdf_call_counter, tf.Variable(2))


def test_cached_integrate_equals_integrate_without_cache():
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CachedPDF(gauss)

    assert tf.math.reduce_all(
        tf.equal(gauss.integrate(limits=obs), cached_gauss.integrate(limits=obs))
    )


def test_integrate_cache_is_used():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(0))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))


def test_integrate_cache_is_revaluation_if_mu_was_changed():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(0))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))
    mu.set_value(2.0)
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))


def test_integrate_cache_is_revaluation_if_sigma_was_changed():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(0))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))
    sigma.set_value(3.0)
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))


def test_integrate_cache_is_revaluation_if_limits_is_different():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(0))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))
    obs = zfit.Space("x", limits=[-5.0, 0.0])
    cached_test_pdf.integrate(obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))
    cached_test_pdf.integrate(obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))


def test_gradient_cached_pdf():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)
    x = znp.linspace(-5, 5, 500)
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:

        pdf = cached_test_pdf.pdf(x)

    with pytest.raises(AnalyticGradientNotAvailable):
        _ = tape.gradient(pdf, [mu])


def test_minimize_cached_pdf():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf1 = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf1)

    mu2 = zfit.Parameter("mu2", 5.0, -5, 5)
    sigma2 = zfit.Parameter("sigma2", 3, 0, 10)
    test_pdf2 = TestPDF(obs=obs, mu=mu2, sigma=sigma2)
    cached_test_pdf2 = CachedPDF(test_pdf2)

    testpdf = zfit.pdf.SumPDF([cached_test_pdf, cached_test_pdf2], fracs=0.5)

    array1 = np.random.normal(1.3, 0.8, 5000)
    array2 = np.random.normal(3.4, 1.3, 5000)
    array = znp.concatenate([array1, array2])
    data = zfit.Data.from_numpy(obs=obs, array=array)
    nll = zfit.loss.UnbinnedNLL(model=testpdf, data=data)
    minimizer = zfit.minimize.Minuit(gradient="zfit")
    with pytest.raises(AnalyticGradientNotAvailable):
        _ = minimizer.minimize(nll)
    minimizer = zfit.minimize.Minuit(gradient=True)
    result = minimizer.minimize(nll)
    assert result.converged
    assert result.valid
    result.hesse()

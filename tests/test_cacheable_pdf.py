import zfit
import zfit.z.numpy as znp
import tensorflow as tf

from zfit import supports
from zfit.models.cache import CacheablePDF
from zfit.util import ztyping


class TestPDF(zfit.pdf.BaseFunctor):
    def __init__(self, obs: ztyping.ObsTypeInput, mu, sigma):
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        super().__init__(pdfs=gauss, obs=obs)
        self.pdf_call_counter = tf.Variable(0)
        self.integrate_call_counter = tf.Variable(0)

    @supports(norm="space")
    def _pdf(self, x, norm, *, norm_range=None):
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
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)

    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), cached_gauss.pdf(x)))


def test_pdf_cache_is_used():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_gauss = CacheablePDF(gauss)

    assert tf.math.reduce_all(
        tf.equal(gauss.integrate(limits=obs), cached_gauss.integrate(limits=obs))
    )


def test_integrate_cache_is_used():
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
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
    cached_test_pdf = CacheablePDF(test_pdf)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(0))
    cached_test_pdf.integrate(limits=obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(1))
    obs = zfit.Space("x", limits=[-5.0, 0.0])
    cached_test_pdf.integrate(obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))
    cached_test_pdf.integrate(obs)
    assert tf.equal(test_pdf.integrate_call_counter, tf.Variable(2))

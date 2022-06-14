import zfit
import zfit.z.numpy as znp
import tensorflow as tf
from zfit.models.cache import CacheablePDF


def test_cached_pdf_equals_pdf_without_cache():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)

    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), cached_gauss.pdf(x)))


def test_pdf_cache_is_used():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)
    cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(0.))
    cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(1.))


def test_pdf_cache_revaluation_if_mu_was_changed():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)
    cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(0.))
    mu.set_value(3.)
    pdf = cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(0.))
    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), pdf))


def test_pdf_cache_revaluation_if_sigma_was_changed():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)
    cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(0.))
    sigma.set_value(2.)
    pdf = cached_gauss.pdf(x)
    assert tf.equal(cached_gauss.pdf_cache_counter, tf.Variable(0.))
    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), pdf))


def test_cached_integrate_equals_integrate_without_cache():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)

    assert tf.math.reduce_all(tf.equal(gauss.integrate(limits=obs), cached_gauss.integrate(limits=obs)))


def test_integrate_cache_is_used():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(1.))


def test_integrate_cache_is_revaluation_if_mu_was_changed():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    mu.set_value(3.)
    integral = cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    assert tf.math.reduce_all(tf.equal(gauss.integrate(limits=obs), integral))


def test_integrate_cache_is_revaluation_if_sigma_was_changed():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    sigma.set_value(2.)
    integral = cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    assert tf.math.reduce_all(tf.equal(gauss.integrate(limits=obs), integral))


def test_integrate_cache_is_revaluation_if_limits_is_different():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    cached_gauss.integrate(limits=obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    obs = zfit.Space('x', limits=[-5., 0.])
    integral = cached_gauss.integrate(obs)
    assert tf.equal(cached_gauss.integrate_cache_counter, tf.Variable(0.))
    assert tf.math.reduce_all(tf.equal(gauss.integrate(limits=obs), integral))

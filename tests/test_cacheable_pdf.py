import zfit
import zfit.z.numpy as znp
import tensorflow as tf
from zfit.models.cache import CacheablePDF


def test_pdf():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)
    x = znp.linspace(-5, 5, 500)

    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), cached_gauss.pdf(x)))


def test_integrate():
    mu = zfit.Parameter('mu', 1., -5, 5)
    sigma = zfit.Parameter('sigma', 1, 0, 10)
    obs = zfit.Space('x', limits=[-5., 5.])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CacheablePDF(gauss)

    assert tf.math.reduce_all(tf.equal(gauss.integrate(limits=obs), cached_gauss.integrate(limits=obs)))

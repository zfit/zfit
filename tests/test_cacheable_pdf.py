import pytest
import zfit
import zfit.z.numpy as znp
import tensorflow as tf
from zfit.models.cache import CacheablePDF

mu = zfit.Parameter("mu", 1.0, -5, 5)
sigma = zfit.Parameter("sigma", 1, 0, 10)
obs = zfit.Space("x", limits=[-5.0, 5.0])
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
cached_gauss = CacheablePDF(gauss)
x = znp.linspace(-5, 5, 500)


def test_pdf():
    assert tf.math.reduce_all(tf.equal(gauss.pdf(x), cached_gauss.pdf(x)))


def test_integrate():
    assert tf.math.reduce_all(
        tf.equal(gauss.integrate(limits=obs), cached_gauss.integrate(limits=obs))
    )

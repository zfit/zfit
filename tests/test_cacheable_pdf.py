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




def test_minimize_cached_pdf_with_numerical_gradients():
    """Test minimization with cached PDFs using numerical gradients."""
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf1 = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf1)

    mu2 = zfit.Parameter("mu2", 5.0, -5, 10)
    sigma2 = zfit.Parameter("sigma2", 3, 0, 10)
    test_pdf2 = TestPDF(obs=obs, mu=mu2, sigma=sigma2)
    cached_test_pdf2 = CachedPDF(test_pdf2)

    testpdf = zfit.pdf.SumPDF([cached_test_pdf, cached_test_pdf2], fracs=0.5)

    array1 = np.random.normal(1.3, 0.8, 100)  # Smaller dataset for faster testing
    array2 = np.random.normal(3.4, 1.3, 100)
    array = znp.concatenate([array1, array2])
    data = zfit.Data.from_numpy(obs=obs, array=array)
    nll = zfit.loss.UnbinnedNLL(model=testpdf, data=data)

    # Use numerical gradients since analytic gradients are not supported for cached PDFs
    minimizer = zfit.minimize.Minuit(gradient=True)
    result = minimizer.minimize(nll)
    assert result.converged
    assert result.valid
    result.hesse()


def test_basic_functionality():
    """Test basic functionality of CachedPDF without gradients."""
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    cached_gauss = CachedPDF(gauss)  # analytic_gradients=False by default

    x = znp.linspace(-5, 5, 100)

    # Test basic PDF evaluation
    pdf_val = cached_gauss.pdf(x)
    assert pdf_val is not None
    assert pdf_val.shape == (100,)

    # Test integration
    integral = cached_gauss.integrate(limits=obs)
    assert integral is not None
    assert znp.allclose(integral, 1.0, atol=1e-6)  # Should be normalized


def test_cached_pdf_performance():
    """Test that caching actually improves performance."""
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)
    cached_test_pdf = CachedPDF(test_pdf)

    x = znp.linspace(-5, 5, 1000)

    # Reset counters
    test_pdf.pdf_call_counter.assign(0)
    test_pdf.integrate_call_counter.assign(0)

    # Multiple calls with same parameters should use cache
    for _ in range(5):
        cached_test_pdf.pdf(x)

    # Should only call underlying function once
    assert test_pdf.pdf_call_counter.numpy() == 1

    # Change parameter and call again
    mu.set_value(2.0)
    cached_test_pdf.pdf(x)

    # Should call underlying function again
    assert test_pdf.pdf_call_counter.numpy() == 2

    # Multiple calls with new parameters should use cache
    for _ in range(3):
        cached_test_pdf.pdf(x)

    # Should still be 2 calls total
    assert test_pdf.pdf_call_counter.numpy() == 2


def test_cached_pdf_with_different_cache_tolerances():
    """Test CachedPDF with different cache tolerance values."""
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    test_pdf = TestPDF(obs=obs, mu=mu, sigma=sigma)

    # Test with very tight tolerance
    cached_tight = CachedPDF(test_pdf, epsilon=1e-12)
    assert cached_tight._cache_tolerance == 1e-12

    # Test with loose tolerance
    cached_loose = CachedPDF(test_pdf, epsilon=1e-4)
    assert cached_loose._cache_tolerance == 1e-4

    # Test default tolerance
    cached_default = CachedPDF(test_pdf)
    assert cached_default._cache_tolerance == 1e-8


def test_to_cached_method():
    """Test the to_cached method on BasePDF."""
    obs = zfit.Space("x", limits=[-5.0, 5.0])
    mu = zfit.Parameter("mu", 1.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1, 0, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Test basic functionality
    cached_gauss = gauss.to_cached()
    assert isinstance(cached_gauss, CachedPDF)
    assert cached_gauss.name == "Gauss_cached"
    assert cached_gauss.obs == gauss.obs
    assert cached_gauss.norm == gauss.norm

    # Test with custom parameters
    cached_gauss_custom = gauss.to_cached(
        epsilon=1e-6,
        name="custom_cached"
    )
    assert cached_gauss_custom.name == "custom_cached"
    assert cached_gauss_custom._cache_tolerance == 1e-6

    # Test default parameters
    cached_gauss_default = gauss.to_cached()
    assert cached_gauss_default._cache_tolerance == 1e-8

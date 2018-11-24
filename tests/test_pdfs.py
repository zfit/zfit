import pytest
import tensorflow as tf
import numpy as np

from zfit import ztf
from zfit.core.limits import Range
from zfit.core.parameter import Parameter
from zfit.models.functor import SumPDF, ProductPDF
from zfit.models.basic import Gauss
import zfit

low, high = -0.64, 5.9


def true_gaussian_sum(x):
    sum_gauss = 0.3 * np.exp(- (x - 1.) ** 2 / (2 * 11. ** 2))
    sum_gauss += 0.2 * np.exp(- (x - 2.) ** 2 / (2 * 22. ** 2))
    sum_gauss += 0.5 * np.exp(- (x - 3.) ** 2 / (2 * 33. ** 2))
    return sum_gauss


# @pytest.fixture()
def sum_prod_gauss():
    mu1 = Parameter("mu1a", 1.)
    mu2 = Parameter("mu2a", 2.)
    mu3 = Parameter("mu3a", 3.)
    sigma1 = Parameter("sigma1a", 11.)
    sigma2 = Parameter("sigma2a", 22.)
    sigma3 = Parameter("sigma3a", 33.)
    gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss1a")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss2a")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss3a")
    zfit.sess.run(tf.global_variables_initializer())
    gauss_dists = [gauss1, gauss2, gauss3]
    sum_gauss = SumPDF(pdfs=gauss_dists, fracs=[0.3, 0.15])
    prod_gauss = ProductPDF(pdfs=gauss_dists, dims=[(0,), (0,), (0,)])
    prod_gauss_3d = ProductPDF(pdfs=gauss_dists, dims=[(0,), (1,), (2,)])
    return sum_gauss, prod_gauss, prod_gauss_3d, gauss_dists


sum_gauss, prod_gauss, prod_gauss_3d, gauss_dists = sum_prod_gauss()


# with tf.Session() as sess:


# init = tf.global_variables_initializer()
# sess.run(init)

def test_prod_gauss_nd():
    # return
    test_values = np.random.random(size=(3, 10))
    probs = prod_gauss_3d.pdf(x= test_values, norm_range=(-5, 4))
    zfit.sess.run(tf.global_variables_initializer())
    true_probs = np.prod([gauss.pdf(test_values[i,:], norm_range=(-5, 4)) for i, gauss in enumerate(gauss_dists)])
    probs_np = zfit.sess.run(probs)
    print(probs_np)
    np.testing.assert_allclose(zfit.sess.run(true_probs), probs_np[0,:], rtol=1e-2)


def test_func_sum():
    zfit.sess.run(tf.global_variables_initializer())
    test_values = np.array([3., 12., -0.2, -7.2])
    vals = sum_gauss.as_func(norm_range=False).value(
        x=ztf.convert_to_tensor(test_values, dtype=zfit.settings.types.float))
    vals = zfit.sess.run(vals)
    # test_sum = sum([g.func(test_values) for g in gauss_dists])
    np.testing.assert_allclose(vals, true_gaussian_sum(test_values), rtol=1e-2)  # MC integral


def test_normalization_sum_gauss():
    normalization_testing(sum_gauss)


def test_normalization_sum_gauss_extended():
    test_yield = 109.
    sum_gauss.set_yield(test_yield)
    normalization_testing(sum_gauss, normalization_value=test_yield)


def test_normalization_prod_gauss():
    normalization_testing(prod_gauss)


def normalization_testing(pdf, normalization_value=1.):
    init = tf.global_variables_initializer()
    zfit.sess.run(init)
    with pdf.temp_norm_range(Range.from_boundaries(low, high, dims=Range.FULL)):
        samples = tf.cast(np.random.uniform(low=low, high=high, size=40000),
                          dtype=tf.float64)
        samples.limits = low, high
        probs = pdf.pdf(samples)
        result = zfit.sess.run(probs)
        result = np.average(result) * (high - low)
        assert normalization_value == pytest.approx(result, rel=0.07)


def test_extended_gauss():
    # return  # HACK: no clue whatsoever why this fails...
    with tf.name_scope("gauss_params2"):
        mu1 = Parameter("mu11", 1.)
        mu2 = Parameter("mu21", 12.)
        mu3 = Parameter("mu31", 3.)
        sigma1 = Parameter("sigma11", 1.)
        sigma2 = Parameter("sigma21", 12.)
        sigma3 = Parameter("sigma31", 33.)
        yield1 = Parameter("yield11", 150.)
        yield2 = Parameter("yield21", 550.)
        yield3 = Parameter("yield31", 2500.)
        sum_yields = 150 + 550 + 2500

        gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss11")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss21")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss31")
        gauss1.set_yield(yield1)
        gauss2.set_yield(yield2)
        gauss3.set_yield(yield3)

        gauss_dists = [gauss1, gauss2, gauss3]

        sum_gauss = SumPDF(pdfs=gauss_dists)

    zfit.sess.run(tf.global_variables_initializer())
    normalization_testing(pdf=sum_gauss, normalization_value=sum_yields)


if __name__ == '__main__':
    test_extended_gauss()

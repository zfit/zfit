import pytest
import tensorflow as tf
import numpy as np

from zfit import ztf
from zfit.core.limits import Range
from zfit.core.parameter import Parameter
from zfit.models.functor import SumPDF, ProductPDF
from zfit.models.basic import Gauss
from zfit.models.dist_tfp import Normal
import zfit

low, high = -0.64, 5.9

mu1_true = 1.
mu2_true = 2.
mu3_true = 0.6
sigma1_true = 1.4
sigma2_true = 2.3
sigma3_true = 1.8

fracs = [0.3, 0.15]


def true_gaussian_sum(x):
    sum_gauss = fracs[0] * np.exp(- (x - mu1_true) ** 2 / (2 * sigma1_true ** 2))
    sum_gauss += fracs[1] * np.exp(- (x - mu2_true) ** 2 / (2 * sigma2_true ** 2))
    sum_gauss += (1. - sum(fracs)) * np.exp(- (x - mu3_true) ** 2 / (2 * sigma3_true ** 2))
    return sum_gauss


# @pytest.fixture()
def sum_prod_gauss():
    # define parameters
    mu1 = Parameter("mu1a", mu1_true)
    mu2 = Parameter("mu2a", mu2_true)
    mu3 = Parameter("mu3a", mu3_true)
    sigma1 = Parameter("sigma1a", sigma1_true)
    sigma2 = Parameter("sigma2a", sigma2_true)
    sigma3 = Parameter("sigma3a", sigma3_true)

    # Gauss for sum, same axes
    gauss1 = Gauss(mu=mu1, sigma=sigma1, dims=0, name="gauss1asum")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, dims=0, name="gauss2asum")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, dims=0, name="gauss3asum")
    gauss_dists = [gauss1, gauss2, gauss3]

    zfit.sess.run(tf.global_variables_initializer())
    sum_gauss = SumPDF(pdfs=gauss_dists, fracs=fracs)
    prod_gauss = ProductPDF(pdfs=gauss_dists, dims=(0,))

    # Gauss for product, independent
    gauss13 = Gauss(mu=mu1, sigma=sigma1, dims=0, name="gauss1a")
    gauss23 = Gauss(mu=mu2, sigma=sigma2, dims=1, name="gauss2a")
    gauss33 = Gauss(mu=mu3, sigma=sigma3, dims=2, name="gauss3a")
    gauss_dists3 = [gauss13, gauss23, gauss33]
    prod_gauss_3d = ProductPDF(pdfs=gauss_dists3, dims=range(3))
    prod_gauss_3d.set_integration_options(mc_options={'draws_per_dim': 33})

    gauss12 = Gauss(mu=mu1, sigma=sigma1, dims=3, name="gauss12a")
    gauss22 = Gauss(mu=mu2, sigma=sigma2, dims=0, name="gauss22a")
    gauss32 = Gauss(mu=mu3, sigma=sigma3, dims=2, name="gauss32a")
    gauss_dists2 = [gauss12, gauss22, gauss32]
    prod_gauss_4d = ProductPDF(pdfs=gauss_dists2 + [prod_gauss_3d], dims=range(4))
    prod_gauss_4d.set_integration_options(mc_options={'draws_per_dim': 33})
    return sum_gauss, prod_gauss, prod_gauss_3d, prod_gauss_4d, gauss_dists3, gauss_dists2, gauss_dists


sum_gauss, prod_gauss, prod_gauss_3d, prod_gauss_4d, gauss_dists3, gauss_dists2, gauss_dists = sum_prod_gauss()


# with tf.Session() as sess:


# init = tf.global_variables_initializer()
# sess.run(init)

def test_prod_gauss_nd():
    # return
    test_values = np.random.random(size=(3, 10))
    probs = prod_gauss_3d.pdf(x=test_values, norm_range=(-5, 4))
    zfit.sess.run(tf.global_variables_initializer())
    true_probs = np.prod([gauss.pdf(test_values[i, :], norm_range=(-5, 4)) for i, gauss in enumerate(gauss_dists)])
    probs_np = zfit.sess.run(probs)
    np.testing.assert_allclose(zfit.sess.run(true_probs), probs_np[0, :], rtol=1e-2)


def test_prod_gauss_nd_mixed():
    # return
    test_values = np.random.random(size=(4, 10))
    norm_range = (-5, 4)

    probs = prod_gauss_4d.pdf(x=test_values, norm_range=Range.from_boundaries(lower=(-5,) * 4,
                                                                              upper=(4,) * 4, axes=tuple(range(4))))
    zfit.sess.run(tf.global_variables_initializer())
    gauss1, gauss2, gauss3 = gauss_dists2
    true_probs = [gauss1.pdf(test_values[3, :], norm_range=norm_range)]
    true_probs += [gauss2.pdf(test_values[0, :], norm_range=norm_range)]
    true_probs += [gauss3.pdf(test_values[2, :], norm_range=norm_range)]
    true_probs += [prod_gauss_3d.pdf(test_values[(0, 1, 2), :], norm_range=Range.from_boundaries(lower=(-5,) * 3,
                                                                                                 upper=(4,) * 3,
                                                                                                 axes=tuple(range(3))))]

    true_probs = np.prod([true_probs])
    probs_np = zfit.sess.run(probs)
    np.testing.assert_allclose(zfit.sess.run(true_probs), probs_np[0, :], rtol=1e-2)


def test_func_sum():
    zfit.sess.run(tf.global_variables_initializer())
    test_values = np.random.uniform(low=-3, high=4, size=10)
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
    with pdf.temp_norm_range(Range.from_boundaries(low, high, axes=Range.FULL)):
        samples = tf.cast(np.random.uniform(low=low, high=high, size=(pdf.n_dims, 40000)),
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

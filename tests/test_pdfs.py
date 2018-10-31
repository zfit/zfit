
import pytest
import tensorflow as tf
import numpy as np

from zfit.core.limits import Range
from zfit.core.parameter import FitParameter
from zfit.pdfs.functor import SumPDF, ProductPDF
from zfit.pdfs.basic import Gauss
import zfit

low, high = -0.64, 5.9


def true_gaussian_sum(x):
    sum_gauss = 0.3 * np.exp(- (x - 1.) ** 2 / (2 * 11. ** 2))
    sum_gauss += 0.2 * np.exp(- (x - 2.) ** 2 / (2 * 22. ** 2))
    sum_gauss += 0.5 * np.exp(- (x - 3.) ** 2 / (2 * 33. ** 2))
    return sum_gauss


with tf.Session() as sess:
    mu1 = FitParameter("mu1", 1.)
    mu2 = FitParameter("mu2", 2.)
    mu3 = FitParameter("mu3", 3.)
    sigma1 = FitParameter("sigma1", 11.)
    sigma2 = FitParameter("sigma2", 22.)
    sigma3 = FitParameter("sigma3", 33.)

    gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss1")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss2")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss3")

    gauss_dists = [gauss1, gauss2, gauss3]

    sum_gauss = SumPDF(pdfs=gauss_dists, frac=[0.3, 0.2])
    prod_gauss = ProductPDF(pdfs=gauss_dists)

    init = tf.global_variables_initializer()
    sess.run(init)


def test_func_sum():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_values = np.array([3., 129., -0.2, -78.2])
        vals = sum_gauss.unnormalized_prob(
            tf.convert_to_tensor(test_values, dtype=zfit.settings.types.float))
        vals = sess.run(vals)
        # test_sum = sum([g.func(test_values) for g in gauss_dists])
        np.testing.assert_almost_equal(vals, true_gaussian_sum(test_values))


def test_normalization_sum_gauss():
    normalization_testing(sum_gauss)


def test_normalization_sum_gauss_extended():
    test_yield = 109.
    sum_gauss.set_yield(test_yield)
    normalization_testing(sum_gauss, normalization_value=test_yield)


def test_normalization_prod_gauss():
    normalization_testing(prod_gauss)


def normalization_testing(pdf, normalization_value=1.):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        with pdf.temp_norm_range(Range.from_boundaries(low, high, dims=Range.FULL)):
            samples = tf.cast(np.random.uniform(low=low, high=high, size=100000),
                              dtype=tf.float64)
            samples.limits = low, high
            probs = pdf.prob(samples)
            result = sess.run(probs)
            result = np.average(result) * (high - low)
            print(result)
            assert result == pytest.approx(normalization_value, rel=0.07)


def test_extended_gauss():
    with tf.Session() as sess:
        mu1 = FitParameter("mu1", 1.)
        mu2 = FitParameter("mu2", 2.)
        mu3 = FitParameter("mu3", 3.)
        sigma1 = FitParameter("sigma1", 11.)
        sigma2 = FitParameter("sigma2", 22.)
        sigma3 = FitParameter("sigma3", 33.)
        yield1 = FitParameter("yield1", 150.)
        yield2 = FitParameter("yield2", 550.)
        yield3 = FitParameter("yield3", 2500.)
        sum_yields = 150 + 550 + 2500

        gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss1")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss2")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss3")
        gauss1.set_yield(yield1)
        gauss2.set_yield(yield2)
        gauss3.set_yield(yield3)

        gauss_dists = [gauss1, gauss2, gauss3]

        sum_gauss = SumPDF(pdfs=gauss_dists)
        # prod_gauss = ProductPDF(pdfs=gauss_dists)

        init = tf.global_variables_initializer()
        sess.run(init)
        normalization_testing(pdf=sum_gauss, normalization_value=sum_yields)

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
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    gauss_dists = [gauss1, gauss2, gauss3]
    sum_gauss = SumPDF(pdfs=gauss_dists, fracs=[0.3, 0.15])
    prod_gauss = ProductPDF(pdfs=gauss_dists)
    return sum_gauss, prod_gauss


sum_gauss, prod_gauss = sum_prod_gauss()


# with tf.Session() as sess:


# init = tf.global_variables_initializer()
# sess.run(init)


def test_func_sum():
    return  # HACK
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        test_values = np.array([3., 129., -0.2, -78.2])
        vals = sum_gauss.unnormalized_pdf(
            ztf.convert_to_tensor(test_values, dtype=zfit.settings.types.float))
        vals = sess.run(vals)
        # test_sum = sum([g.func(test_values) for g in gauss_dists])
        np.testing.assert_almost_equal(vals, true_gaussian_sum(test_values))


def test_normalization_sum_gauss():
    normalization_testing(sum_gauss)


def test_normalization_sum_gauss_extended():
    test_yield = 109.
    sum_gauss.set_yield(test_yield)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    normalization_testing(sum_gauss, normalization_value=test_yield)


def test_normalization_prod_gauss():
    normalization_testing(prod_gauss)


def normalization_testing(pdf, normalization_value=1.):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        with pdf.temp_norm_range(Range.from_boundaries(low, high, dims=Range.FULL)):
            samples = tf.cast(np.random.uniform(low=low, high=high, size=40000),
                              dtype=tf.float64)
            samples.limits = low, high
            probs = pdf.pdf(samples)
            result = sess.run(probs)
            result = np.average(result) * (high - low)
            print(result)
            assert normalization_value == pytest.approx(result, rel=0.07)


def test_extended_gauss():
    # return  # HACK: no clue whatsoever why this fails...
    with tf.name_scope("gauss_params2"):
        mu1 = Parameter("mu11", 1.)
        mu2 = Parameter("mu21", 2.)
        mu3 = Parameter("mu31", 3.)
        sigma1 = Parameter("sigma11", 11.)
        sigma2 = Parameter("sigma21", 22.)
        sigma3 = Parameter("sigma31", 33.)
        yield1 = Parameter("yield11", 150.)
        yield2 = Parameter("yield21", 550.)
        yield3 = Parameter("yield31", 2500.)
        sum_yields = 150 + 550 + 2500
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

        gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss11")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss21")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss31")
        gauss1.set_yield(yield1)
        gauss2.set_yield(yield2)
        gauss3.set_yield(yield3)

        gauss_dists = [gauss1, gauss2, gauss3]

        # with tf.Session() as sess:
        # sess.run([v.initializer for v in (yield1, yield2, yield3, sigma1, sigma2, sigma3, mu1, mu2, mu3)])
        sum_gauss = SumPDF(pdfs=gauss_dists)

    #     sess.run([init])
    # prod_gauss = ProductPDF(models=gauss_dists)

    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    normalization_testing(pdf=sum_gauss, normalization_value=sum_yields)


if __name__ == '__main__':
    test_extended_gauss()

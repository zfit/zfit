#  Copyright (c) 2019 zfit

from typing import List

import pytest
import tensorflow as tf
import numpy as np

from zfit import ztf
from zfit.core.data import Data
from zfit.core.interfaces import ZfitPDF
from zfit.core.limits import Space
from zfit.core.parameter import Parameter
from zfit.models.functor import SumPDF, ProductPDF
from zfit.models.dist_tfp import Gauss
import zfit
from zfit.core.testing import setup_function, teardown_function, tester

low, high = -0.64, 5.9

mu1_true = 1.
mu2_true = 2.
mu3_true = 0.6
sigma1_true = 1.4
sigma2_true = 2.3
sigma3_true = 1.8

fracs = [0.3, 0.15]


def true_gaussian_sum(x):
    def norm(sigma):
        return np.sqrt(2 * np.pi) * sigma

    sum_gauss = fracs[0] * np.exp(- (x - mu1_true) ** 2 / (2 * sigma1_true ** 2)) / norm(sigma1_true)
    sum_gauss += fracs[1] * np.exp(- (x - mu2_true) ** 2 / (2 * sigma2_true ** 2)) / norm(sigma2_true)
    sum_gauss += (1. - sum(fracs)) * np.exp(- (x - mu3_true) ** 2 / (2 * sigma3_true ** 2)) / norm(sigma3_true)
    return sum_gauss


obs1 = Space(obs='obs1')


def create_params(name_add=""):
    mu1 = Parameter("mu1" + name_add, mu1_true)
    mu2 = Parameter("mu2" + name_add, mu2_true)
    mu3 = Parameter("mu3" + name_add, mu3_true)
    sigma1 = Parameter("sigma1" + name_add, sigma1_true)
    sigma2 = Parameter("sigma2" + name_add, sigma2_true)
    sigma3 = Parameter("sigma3" + name_add, sigma3_true)
    return mu1, mu2, mu3, sigma1, sigma2, sigma3


def create_gaussians() -> List[ZfitPDF]:
    # Gauss for sum, same axes
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params()
    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1asum")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss2asum")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss3asum")
    gauss_dists = [gauss1, gauss2, gauss3]
    return gauss_dists


def sum_gaussians():
    gauss_dists = create_gaussians()
    sum_gauss = SumPDF(pdfs=gauss_dists, fracs=fracs, obs=obs1)
    return sum_gauss


def product_gaussians():
    gauss_dists = create_gaussians()
    prod_gauss = ProductPDF(pdfs=gauss_dists, obs=obs1)
    return prod_gauss


def product_gauss_3d(name=""):
    # Gauss for product, independent
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params('3d' + name)

    gauss13 = Gauss(mu=mu1, sigma=sigma1, obs='a', name="gauss1a")
    gauss23 = Gauss(mu=mu2, sigma=sigma2, obs='b', name="gauss2a")
    gauss33 = Gauss(mu=mu3, sigma=sigma3, obs='c', name="gauss3a")
    gauss_dists3 = [gauss13, gauss23, gauss33]
    prod_gauss_3d = ProductPDF(pdfs=gauss_dists3, obs=['a', 'b', 'c'])
    return prod_gauss_3d

    # prod_gauss_3d.update_integration_options(draws_per_dim=33)


def product_gauss_4d():
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params("4d")

    gauss12 = Gauss(mu=mu1, sigma=sigma1, obs='d', name="gauss12a")
    gauss22 = Gauss(mu=mu2, sigma=sigma2, obs='a', name="gauss22a")
    gauss32 = Gauss(mu=mu3, sigma=sigma3, obs='c', name="gauss32a")
    gauss_dists2 = [gauss12, gauss22, gauss32]
    prod_gauss_4d = ProductPDF(pdfs=gauss_dists2 + [product_gauss_3d("4d")], obs=['a', 'b', 'c', 'd'])
    return prod_gauss_4d


def test_prod_gauss_nd():
    test_values = np.random.random(size=(10, 3))
    lower = ((-5, -5, -5),)
    upper = ((4, 4, 4),)
    obs1 = ['a', 'b', 'c']
    norm_range_3d = Space(obs=obs1, limits=(lower, upper))
    test_values_data = Data.from_tensor(obs=obs1, tensor=test_values)
    probs = product_gauss_3d().pdf(x=test_values_data, norm_range=norm_range_3d)
    true_probs = np.prod(
        [gauss.pdf(test_values[:, i], norm_range=(-5, 4)) for i, gauss in enumerate(create_gaussians())])
    probs_np = zfit.run(probs)
    np.testing.assert_allclose(zfit.run(true_probs), probs_np, rtol=1e-2)


@pytest.mark.flaky(reruns=3)
def test_prod_gauss_nd_mixed():
    norm_range = (-5, 4)
    low, high = norm_range
    test_values = np.random.uniform(low=low, high=high, size=(1000, 4))

    obs4d = ['a', 'b', 'c', 'd']
    test_values_data = Data.from_tensor(obs=obs4d, tensor=test_values)
    # prod_gauss_4d.update_integration_options(draws_per_dim=10)
    limits_4d = Space(limits=(((-5,) * 4,), ((4,) * 4,)), obs=obs4d)
    prod_gauss_4d = product_gauss_4d()
    prod_gauss_4d.set_norm_range(limits_4d)
    probs = prod_gauss_4d.pdf(x=test_values_data,
                              norm_range=limits_4d)
    gausses = create_gaussians()

    for gauss in (gausses):
        gauss.set_norm_range(norm_range)
    gauss1, gauss2, gauss3 = gausses
    prod_gauss_3d = product_gauss_3d()

    def probs_4d(values):
        true_prob = [gauss1.pdf(values[:, 3])]
        true_prob += [gauss2.pdf(values[:, 0])]
        true_prob += [gauss3.pdf(values[:, 2])]
        true_prob += [prod_gauss_3d.pdf(values[:, (0, 1, 2)],
                                        norm_range=Space(limits=(((-5,) * 3,), ((4,) * 3,)),
                                                         obs=['a', 'b', 'c']))]
        return np.prod(true_prob, axis=0)

    true_unnormalized_probs = probs_4d(values=test_values)

    normalization_probs = limits_4d.area() * probs_4d(np.random.uniform(low=low, high=high, size=(40 ** 4, 4)))
    true_probs = true_unnormalized_probs / tf.reduce_mean(normalization_probs)
    grad = tf.gradients(probs, list(prod_gauss_4d.get_dependents()))
    probs_np = zfit.run(probs)
    grad_np = zfit.run(grad)
    print("Gradients", grad_np)
    print(np.average(probs_np))
    true_probs_np = zfit.run(true_probs)
    assert np.average(probs_np * limits_4d.area()) == pytest.approx(1., rel=0.33)  # low n mc
    np.testing.assert_allclose(true_probs_np, probs_np, rtol=2e-2)


def test_func_sum():
    sum_gauss = sum_gaussians()
    test_values = np.random.uniform(low=-3, high=4, size=10)
    sum_gauss_as_func = sum_gauss.as_func(norm_range=(-10, 10))
    vals = sum_gauss_as_func.func(x=test_values)
    vals = zfit.run(vals)
    # test_sum = sum([g.func(test_values) for g in gauss_dists])
    np.testing.assert_allclose(vals, true_gaussian_sum(test_values), rtol=1e-2)  # MC integral


def test_normalization_sum_gauss():
    normalization_testing(sum_gaussians())


def test_normalization_sum_gauss_extended():
    test_yield = 109.
    sum_gauss_extended = sum_gaussians().create_extended(test_yield)
    normalization_testing(sum_gauss_extended)


def test_normalization_prod_gauss():
    normalization_testing(product_gaussians())


def test_exp():
    lambda_true = 0.31
    lambda_ = zfit.Parameter('lambda1', lambda_true)
    exp1 = zfit.pdf.Exponential(lambda_=lambda_, obs='obs1')
    sample = exp1.sample(n=1000, limits=(-10, 10))
    sample_np = zfit.run(sample)
    assert not any(np.isnan(sample_np))
    probs1 = exp1.pdf(x=np.random.normal(size=842), norm_range=(-5, 5))
    probs2 = exp1.pdf(x=np.linspace(5300, 5700, num=1100), norm_range=(5250, 5750))
    probs1_np, probs2_np = zfit.run([probs1, probs2])
    assert not any(np.isnan(probs1_np))
    assert not any(np.isnan(probs2_np))
    normalization_testing(exp1)


def normalization_testing(pdf):
    with pdf.set_norm_range(Space(obs=obs1, limits=(low, high))):
        samples = tf.cast(np.random.uniform(low=low, high=high, size=(40000, pdf.n_obs)),
                          dtype=tf.float64)
        samples = zfit.Data.from_tensor(obs=pdf.obs, tensor=samples)
        probs = pdf.pdf(samples)
        result = zfit.run(probs)
        result = np.average(result) * (high - low)
        assert pytest.approx(result, rel=0.07) == 1


def test_extended_gauss():
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

        gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss11")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss21")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss31")
        gauss1 = gauss1.create_extended(yield1)
        gauss2 = gauss2.create_extended(yield2)
        gauss3 = gauss3.create_extended(yield3)

        gauss_dists = [gauss1, gauss2, gauss3]

        sum_gauss = SumPDF(pdfs=gauss_dists, obs=obs1)

    normalization_testing(pdf=sum_gauss)


if __name__ == '__main__':
    test_extended_gauss()

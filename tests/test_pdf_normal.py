#  Copyright (c) 2019 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import Parameter
from zfit.models.dist_tfp import Gauss
from zfit.core.testing import setup_function, teardown_function, tester

mu1_true = 1.
mu2_true = 2.
mu3_true = 0.6
sigma1_true = 1.4
sigma2_true = 2.3
sigma3_true = 1.8

test_values = np.random.uniform(low=-3, high=5, size=100)
norm_range1 = (-4., 2.)

obs1 = 'obs1'
limits1 = zfit.Space(obs=obs1, limits=(-0.3, 1.5))


def create_gauss():
    mu1 = Parameter("mu1a", mu1_true)
    mu2 = Parameter("mu2a", mu2_true)
    mu3 = Parameter("mu3a", mu3_true)
    sigma1 = Parameter("sigma1a", sigma1_true)
    sigma2 = Parameter("sigma2a", sigma2_true)
    sigma3 = Parameter("sigma3a", sigma3_true)
    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1a")
    normal1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="normal1a")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss2a")
    normal2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="normal2a")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss3a")
    normal3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="normal3a")
    return gauss1, gauss2, gauss3, normal1, normal2, normal3


# gauss1, gauss2, gauss3, normal1, normal2, normal3 = create_gauss()


def test_gauss1():
    gauss1, gauss2, gauss3, normal1, normal2, normal3 = create_gauss()

    probs1 = gauss1.pdf(x=test_values, norm_range=norm_range1)
    probs1_tfp = normal1.pdf(x=test_values, norm_range=norm_range1)
    probs1 = zfit.run(probs1)
    probs1_tfp = zfit.run(probs1_tfp)
    np.testing.assert_allclose(probs1, probs1_tfp, rtol=1e-2)

    probs1_unnorm = gauss1.pdf(x=test_values, norm_range=False)
    probs1_tfp_unnorm = normal1.pdf(x=test_values, norm_range=False)
    probs1_unnorm = zfit.run(probs1_unnorm)
    probs1_tfp_unnorm = zfit.run(probs1_tfp_unnorm)
    assert not np.allclose(probs1_tfp, probs1_tfp_unnorm, rtol=1e-2)
    assert not np.allclose(probs1, probs1_unnorm, rtol=1e-2)
    # np.testing.assert_allclose(probs1_unnorm, probs1_tfp_unnorm, rtol=1e-2)


def test_truncated_gauss():
    high = 2.
    low = -0.5
    truncated_gauss = zfit.pdf.TruncatedGauss(mu=1, sigma=2, low=low, high=high, obs=limits1)
    gauss = zfit.pdf.Gauss(mu=1., sigma=2, obs=limits1)

    probs_truncated = truncated_gauss.pdf(test_values)
    probs_gauss = gauss.pdf(test_values)

    probs_truncated_np, probs_gauss_np = zfit.run([probs_truncated, probs_gauss])

    bool_index_inside = np.logical_and(low < test_values, test_values < high)
    inside_probs_truncated = probs_truncated_np[bool_index_inside]
    outside_probs_truncated = probs_truncated_np[np.logical_not(bool_index_inside)]
    inside_probs_gauss = probs_gauss_np[bool_index_inside]

    assert inside_probs_gauss == pytest.approx(inside_probs_truncated, rel=1e-3)
    assert all(0 == outside_probs_truncated)

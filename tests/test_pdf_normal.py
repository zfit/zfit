#  Copyright (c) 2022 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
from zfit.models.dist_tfp import Gauss

mu1_true = 1.0
mu2_true = 2.0
mu3_true = 0.6
sigma1_true = 1.4
sigma2_true = 2.3
sigma3_true = 1.8

test_values = np.random.uniform(low=-3, high=5, size=100)
norm_range1 = (-4.0, 2.0)

obs1 = "obs1"
limits1 = zfit.Space(obs=obs1, limits=(-0.3, 1.5))


def create_gauss():
    mu1 = Parameter("mu1", mu1_true)
    mu2 = Parameter("mu2", mu2_true)
    mu3 = Parameter("mu3", mu3_true)
    sigma1 = Parameter("sigma1", sigma1_true)
    sigma2 = Parameter("sigma2", sigma2_true)
    sigma3 = Parameter("sigma3", sigma3_true)
    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1")
    normal1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="normal1")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss2")
    normal2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="normal2")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss3")
    normal3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="normal3")
    return gauss1, gauss2, gauss3, normal1, normal2, normal3


# gauss1, gauss2, gauss3, normal1, normal2, normal3 = create_gauss()


def test_gauss1():
    gauss1, gauss2, gauss3, normal1, normal2, normal3 = create_gauss()

    probs1 = gauss1.pdf(x=test_values, norm=norm_range1)
    probs1_tfp = normal1.pdf(x=test_values, norm=norm_range1)
    probs1 = probs1.numpy()
    probs1_tfp = probs1_tfp.numpy()
    np.testing.assert_allclose(probs1, probs1_tfp, rtol=1e-2)

    probs1_unnorm = gauss1.pdf(x=test_values, norm=False)
    probs1_tfp_unnorm = normal1.pdf(x=test_values, norm=False)
    probs1_unnorm = probs1_unnorm.numpy()
    probs1_tfp_unnorm = probs1_tfp_unnorm.numpy()
    assert not np.allclose(probs1_tfp, probs1_tfp_unnorm, rtol=1e-2)
    assert not np.allclose(probs1, probs1_unnorm, rtol=1e-2)
    # np.testing.assert_allclose(probs1_unnorm, probs1_tfp_unnorm, rtol=1e-2)


def test_truncated_gauss():
    high = 2.0
    low = -0.5
    truncated_gauss = zfit.pdf.TruncatedGauss(
        mu=1, sigma=2, low=low, high=high, obs=limits1
    )
    gauss = zfit.pdf.Gauss(mu=1.0, sigma=2, obs=limits1)

    probs_truncated = truncated_gauss.pdf(test_values)
    probs_gauss = gauss.pdf(test_values)

    probs_truncated_np, probs_gauss_np = [probs_truncated.numpy(), probs_gauss.numpy()]

    bool_index_inside = np.logical_and(low < test_values, test_values < high)
    inside_probs_truncated = probs_truncated_np[bool_index_inside]
    outside_probs_truncated = probs_truncated_np[np.logical_not(bool_index_inside)]
    inside_probs_gauss = probs_gauss_np[bool_index_inside]

    assert inside_probs_gauss == pytest.approx(inside_probs_truncated, rel=1e-3)
    assert all(outside_probs_truncated == 0)

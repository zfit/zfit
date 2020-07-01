#  Copyright (c) 2020 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.models.dist_tfp import Gauss

mean1_true = 1.
width1_true = 1.4

test_values = np.random.uniform(low=-3, high=5, size=100)
norm_range1 = (-4., 2.)

obs1 = 'obs1'
limits1 = zfit.Space(obs=obs1, limits=(-0.3, 1.5))


def test_gauss1():
    bw1 = zfit.pdf.
    probs1 = gauss1.pdf(x=test_values, norm_range=norm_range1)
    probs1_tfp = normal1.pdf(x=test_values, norm_range=norm_range1)
    probs1 = probs1.numpy()
    probs1_tfp = probs1_tfp.numpy()
    np.testing.assert_allclose(probs1, probs1_tfp, rtol=1e-2)

    probs1_unnorm = gauss1.pdf(x=test_values, norm_range=False)
    probs1_tfp_unnorm = normal1.pdf(x=test_values, norm_range=False)
    probs1_unnorm = probs1_unnorm.numpy()
    probs1_tfp_unnorm = probs1_tfp_unnorm.numpy()
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

    probs_truncated_np, probs_gauss_np = [probs_truncated.numpy(), probs_gauss.numpy()]

    bool_index_inside = np.logical_and(low < test_values, test_values < high)
    inside_probs_truncated = probs_truncated_np[bool_index_inside]
    outside_probs_truncated = probs_truncated_np[np.logical_not(bool_index_inside)]
    inside_probs_gauss = probs_gauss_np[bool_index_inside]

    assert inside_probs_gauss == pytest.approx(inside_probs_truncated, rel=1e-3)
    assert all(outside_probs_truncated == 0)

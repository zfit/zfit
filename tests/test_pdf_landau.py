#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import pylandau

import zfit
from zfit import z
import zfit.z.numpy as znp
from zfit.core.testing import tester
from zfit.models.physics import Landau

mu = 2.0
sigma = 1.0

def _landau_params_factory(name_add=""):
    mu_ = zfit.Parameter(f"mu_landau{name_add}", mu)
    sigma_ = zfit.Parameter(f"sigma_landau{name_add}", sigma)
    return {"mu": mu_, "sigma": sigma_}

tester.register_pdf(pdf_class=Landau, params_factories=_landau_params_factory)

# test to ensure no nan values
def sample_testing(pdf):
    sample = pdf.sample(n=1000)
    assert not any(np.isnan(sample.value()))

# test to check PDF
def eval_testing(pdf, x): # -> not yet implemented below.
    probs = pdf.pdf(x)
    assert probs.shape.rank == 1
    assert probs.shape[0] == x.shape[0]
    probs = znp.asarray(probs)
    assert not np.any(np.isnan(probs))
    return probs

def ensure_positivity(landau, test_values):
    probs1 = landau.pdf(x=test_values, norm=False)
    np.testing.assert_array_less(0, probs1)
    sample1 = landau.sample(100)
    assert len(sample1.value()) == 100


def test_accuracy_and_sampling(landau, mean, sigma, test_values):
    probs1 = landau.pdf(test_values, norm=False)

    sample1 = landau.sample(100)
    assert len(sample1.value()) == 100

    probs2 = pylandau.landau((test_values - mean) / sigma)
    np.testing.assert_allclose(probs1, probs2, rtol=1e-10, atol=1e-08)

def test_landau():
    mean1_true = 1.0
    std1_true = 2.0

    obs1 = "obs1"
    limits1 = (-10, 10)
    obs1 = zfit.Space(obs1, limits=limits1)
    test_values = z.random.uniform(minval=-10, maxval=10, shape=(1000,))

    mean = _landau_params_factory()["mu"]
    sigma = _landau_params_factory()["sigma"]
    landau1 = zfit.pdf.Landau(mu=mean, sigma=sigma, obs=obs1)

    sample_testing(landau1)

    # ensure_positivity(landau1, test_values)

    # test_accuracy_and_sampling(landau1, mean, sigma, test_values)

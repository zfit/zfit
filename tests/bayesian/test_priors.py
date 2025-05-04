#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit

from zfit._bayesian.priors import (
    NormalPrior,
    UniformPrior,
    HalfNormalPrior,
    GammaPrior,
    BetaPrior,
)


def test_normal_prior():
    prior = NormalPrior(mu=0.0, sigma=1.0)
    samples = prior.sample(1000)
    assert samples.shape[0] == 1000
    assert samples.shape[1] == 1
    assert -5 < np.mean(samples) < 5
    assert 0.5 < np.std(samples) < 1.5


def test_uniform_prior():
    prior = UniformPrior(lower=0.0, upper=1.0)
    samples = prior.sample(1000)
    assert samples.shape[0] == 1000
    assert samples.shape[1] == 1
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert 0.4 < np.mean(samples) < 0.6


def test_half_normal_prior():
    prior = HalfNormalPrior(mu=0.01, sigma=1.0)
    samples = prior.sample(1000)
    assert samples.shape[0] == 1000
    assert samples.shape[1] == 1
    assert np.all(samples >= 0.0)
    assert 0.5 < np.mean(samples) < 1.0


def test_gamma_prior():
    prior = GammaPrior(alpha=2.0, beta=0.5)
    samples = prior.sample(1000)
    assert samples.shape[0] == 1000
    assert samples.shape[1] == 1
    assert np.all(samples >= 0.0)
    assert 0.5 < np.mean(samples) < 1.5


def test_beta_prior():
    prior = BetaPrior(alpha=2.0, beta=2.0)
    samples = prior.sample(1000)
    assert samples.shape[0] == 1000
    assert samples.shape[1] == 1
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert 0.4 < np.mean(samples) < 0.6


def test_add_prior_to_parameter():
    prior = NormalPrior(mu=0.0, sigma=1.0)
    param = zfit.Parameter("test", 1.0, prior=prior)

    param_with_prior = param
    assert param_with_prior.prior == prior
    assert np.isfinite(param_with_prior.prior.log_pdf())


def test_set_priors():
    prior1 = NormalPrior(mu=0.0, sigma=1.0)
    prior2 = UniformPrior(lower=0.0, upper=1.0)
    param1 = zfit.Parameter("test1", 1.0, prior=prior1)
    param2 = zfit.Parameter("test2", 2.0, prior=prior2)

    params_with_priors = [param1, param2]
    assert len(params_with_priors) == 2
    assert params_with_priors[0].prior == prior1
    assert params_with_priors[1].prior == prior2


def test_prior_log_pdf():
    # Test log_pdf for each prior type
    x = np.array([[0.5]])

    priors = [
        NormalPrior(mu=0.0, sigma=1.0),
        UniformPrior(lower=0.0, upper=1.0),
        HalfNormalPrior(mu=0.01, sigma=1.0),
        GammaPrior(alpha=2.0, beta=2.0),
        BetaPrior(alpha=2.0, beta=2.0),
    ]

    for prior in priors:
        log_prob = prior.log_pdf(x)
        assert np.isfinite(log_prob)


def test_prior_sampling():
    # Test sampling with different sample sizes
    n_samples = [1, 10, 100]

    priors = [
        NormalPrior(mu=0.0, sigma=1.0),
        UniformPrior(lower=0.0, upper=1.0),
        HalfNormalPrior(mu=0.01, sigma=1.0),
        GammaPrior(alpha=2.0, beta=2.0),
        BetaPrior(alpha=2.0, beta=2.0),
    ]

    for prior in priors:
        for n in n_samples:
            samples = prior.sample(n)
            assert samples.shape[0] == n
            assert samples.shape[1] == 1

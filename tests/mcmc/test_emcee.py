#  Copyright (c) 2025 zfit


import pytest
import numpy as np
import zfit
from zfit.loss import UnbinnedNLL
from zfit._mcmc.emcee import EmceeSampler


def test_emcee_init():
    """Test EmceeSampler initialization."""
    # Test default initialization
    sampler = EmceeSampler()
    assert sampler.nwalkers is None
    assert sampler.name == "EmceeSampler"

    # Test custom initialization
    sampler = EmceeSampler(nwalkers=50, name="CustomSampler")
    assert sampler.nwalkers == 50
    assert sampler.name == "CustomSampler"


def test_emcee_sampling():
    """Test basic sampling functionality."""
    # Create a simple Gaussian model
    mu = zfit.Parameter("mu", 0.0, prior=zfit.prior.NormalPrior(0.0, 2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10.0, prior=zfit.prior.HalfNormalPrior(0.1, 1.0))

    # Generate some test data
    n_points = 1000
    data = np.random.normal(0, 1, n_points)
    obs = zfit.Space("x", limits=(-5, 5))
    dataset = zfit.Data.from_numpy(obs=obs, array=data)

    # Create model and loss
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    loss = UnbinnedNLL(model=model, data=dataset)

    # Sample
    sampler = EmceeSampler(nwalkers=20)
    n_samples = 100
    n_warmup = 50

    posteriors = sampler.sample(loss=loss, params=[mu, sigma], n_samples=n_samples, n_warmup=n_warmup)

    # Check results
    assert posteriors.samples.shape == (n_samples * 20, 2)  # n_samples * nwalkers, n_params
    assert len(posteriors.param_names) == 2
    assert posteriors.n_warmup == n_warmup
    assert posteriors.n_samples == n_samples


def test_emcee_parameter_limits():
    """Test sampling respects parameter limits."""
    # Create a parameter with limits
    param = zfit.Parameter("test_param", 1.0, 0.0, 2.0, prior=zfit.prior.UniformPrior(0.0, 2.0))

    # Create a simple model and loss
    obs = zfit.Space("x", limits=(-5, 5))
    model = zfit.pdf.Uniform(obs=obs, low=-5, high=param)
    data = zfit.Data.from_numpy(obs=obs, array=np.random.uniform(-5, 5, 100))
    loss = UnbinnedNLL(model=model, data=data)

    # Sample
    sampler = EmceeSampler(nwalkers=10)
    posteriors = sampler.sample(loss=loss, params=[param], n_samples=50, n_warmup=20)

    # Check that all samples are within bounds
    samples = posteriors.samples
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 2.0)


def test_emcee_errors():
    """Test error handling."""
    sampler = EmceeSampler()

    # Test with invalid loss
    with pytest.raises(TypeError):
        sampler.sample(loss="not_a_loss", n_samples=100)

    # Test with invalid parameters
    obs = zfit.Space("x", limits=(-5, 5))
    model = zfit.pdf.Uniform(obs=obs, low=-5, high=5)
    data = zfit.Data.from_numpy(obs=obs, array=np.random.uniform(-5, 5, 100))
    loss = UnbinnedNLL(model=model, data=data)

    with pytest.raises(TypeError):
        sampler.sample(loss=loss, params="not_params", n_samples=100)

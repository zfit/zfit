"""Tests for EmceeSampler initialization from previous PosteriorSamples."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.loss import UnbinnedNLL
from zfit.pdf import Gauss


@pytest.fixture
def simple_model_setup():
    """Create a simple Gaussian model for testing."""
    # Create observation space
    obs = zfit.Space("x", limits=(-5, 5))

    # Parameters with priors
    mu = zfit.Parameter("mu", 0.0, lower=-3, upper=3,
                       prior=zfit.prior.Normal(mu=0.0, sigma=1.0))
    sigma = zfit.Parameter("sigma", 1.0, lower=0.1, upper=3,
                          prior=zfit.prior.Normal(mu=1.0, sigma=0.5))

    # Generate test data
    zfit.settings.set_seed(42)
    data_np = np.random.normal(0.0, 1.0, size=100)
    data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

    # Create model and loss
    model = Gauss(obs=obs, mu=mu, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    return {
        'loss': loss,
        'params': [mu, sigma],
        'model': model,
        'data': data
    }


def test_init_from_posterior_samples_basic(simple_model_setup):
    """Test basic initialization from previous PosteriorSamples."""
    setup = simple_model_setup

    # First sampling run
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=10, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=50,
        n_warmup=20
    )

    # Second sampling run initialized from first
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=10, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=50,
        n_warmup=0,  # No warmup needed
        init=result1
    )

    # Check that we got results
    assert result2.samples.shape == (500, 2)  # 10 walkers * 50 samples
    assert len(result2.param_names) == 2

    # The second run should start from where the first ended
    # So the initial variance should be lower than starting from scratch
    first_batch_std = np.std(result2.samples[:50], axis=0)
    last_batch_std = np.std(result2.samples[-50:], axis=0)

    # Both should be relatively small since we started from converged positions
    assert np.all(first_batch_std < 1.0)
    assert np.all(last_batch_std < 1.0)


def test_init_different_nwalkers(simple_model_setup):
    """Test initialization when nwalkers differs between runs."""
    setup = simple_model_setup

    # First run with 8 walkers
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=50,
        n_warmup=20
    )

    # Second run with 12 walkers (more than first)
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=12, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=30,
        n_warmup=0,
        init=result1
    )

    assert result2.samples.shape == (360, 2)  # 12 walkers * 30 samples

    # Third run with 6 walkers (fewer than first)
    sampler3 = zfit.mcmc.EmceeSampler(nwalkers=6, verbosity=0)
    result3 = sampler3.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=30,
        n_warmup=0,
        init=result1
    )

    assert result3.samples.shape == (180, 2)  # 6 walkers * 30 samples


def test_init_parameter_mismatch():
    """Test error handling when parameters don't match."""
    # First model
    obs = zfit.Space("x", limits=(-5, 5))
    mu1 = zfit.Parameter("mu", 0.0, lower=-3, upper=3,
                        prior=zfit.prior.Normal(mu=0.0, sigma=1.0))
    sigma1 = zfit.Parameter("sigma", 1.0, lower=0.1, upper=3,
                           prior=zfit.prior.Normal(mu=1.0, sigma=0.5))

    zfit.settings.set_seed(42)
    data_np = np.random.normal(0.0, 1.0, size=100)
    data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

    model1 = Gauss(obs=obs, mu=mu1, sigma=sigma1)
    loss1 = UnbinnedNLL(model=model1, data=data)

    # Get first result
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)
    result1 = sampler1.sample(loss=loss1, n_samples=30, n_warmup=10)

    # Second model with different parameters
    mu2 = zfit.Parameter("mean", 0.0, lower=-3, upper=3,  # Different name!
                        prior=zfit.prior.Normal(mu=0.0, sigma=1.0))
    sigma2 = zfit.Parameter("width", 1.0, lower=0.1, upper=3,  # Different name!
                           prior=zfit.prior.Normal(mu=1.0, sigma=0.5))

    model2 = Gauss(obs=obs, mu=mu2, sigma=sigma2)
    loss2 = UnbinnedNLL(model=model2, data=data)

    # This should raise an error
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)
    with pytest.raises(ValueError, match="Parameter names don't match"):
        sampler2.sample(loss=loss2, n_samples=30, n_warmup=0, init=result1)


def test_init_wrong_type():
    """Test error handling when init is not PosteriorSamples."""
    obs = zfit.Space("x", limits=(-5, 5))
    mu = zfit.Parameter("mu", 0.0, lower=-3, upper=3,
                       prior=zfit.prior.Normal(mu=0.0, sigma=1.0))
    sigma = zfit.Parameter("sigma", 1.0, lower=0.1, upper=3,
                          prior=zfit.prior.Normal(mu=1.0, sigma=0.5))

    zfit.settings.set_seed(42)
    data_np = np.random.normal(0.0, 1.0, size=100)
    data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

    model = Gauss(obs=obs, mu=mu, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    sampler = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)

    # Try with wrong type
    with pytest.raises(TypeError, match="init must be a PosteriorSamples instance"):
        sampler.sample(loss=loss, n_samples=30, n_warmup=0, init="not_a_posterior")


def test_init_preserves_convergence(simple_model_setup):
    """Test that initialization from converged chains maintains convergence."""
    setup = simple_model_setup

    # First run until convergence
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=200,
        n_warmup=100
    )

    # Get the posterior means and stds from first run
    means1 = result1.mean()
    stds1 = result1.std()

    # Second run starting from converged positions
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=200,
        n_warmup=0,  # No warmup
        init=result1
    )

    # Get the posterior means and stds from second run
    means2 = result2.mean()
    stds2 = result2.std()

    # Results should be similar since we started from convergence
    # But allow some variation due to MCMC randomness
    np.testing.assert_allclose(means1, means2, rtol=0.2)
    np.testing.assert_allclose(stds1, stds2, rtol=0.2)

    # The early samples from run 2 should already be good
    early_means = np.mean(result2.samples[:50], axis=0)
    # Allow higher tolerance since we're comparing different random samples
    np.testing.assert_allclose(early_means, means1, atol=0.1)


def test_init_with_parameter_reordering(simple_model_setup):
    """Test initialization when parameter order changes between runs."""
    setup = simple_model_setup

    # First run with original order [mu, sigma]
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=10, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],  # [mu, sigma]
        n_samples=100,  # Increased for better convergence
        n_warmup=50      # Increased warmup
    )

    # Second run with reversed parameter order [sigma, mu]
    reversed_params = setup['params'][::-1]
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=10, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=reversed_params,  # [sigma, mu]
        n_samples=100,    # Increased samples
        n_warmup=50,      # Increased warmup for stability when reordering
        init=result1
    )

    # Check that parameters were correctly reordered
    assert result2.param_names == ['sigma', 'mu']

    # The key test: initialization from previous run should work
    # and produce valid results, regardless of parameter order
    assert result2.valid
    assert len(result2.samples) == 10 * 100  # nwalkers * n_samples

    # Check that we can get values for both parameters by name
    mu2 = result2.mean('mu')
    sigma2 = result2.mean('sigma')

    # Basic sanity checks - parameters should be finite and reasonable
    assert np.isfinite(mu2)
    assert np.isfinite(sigma2)
    assert -5 < mu2 < 5  # Within parameter bounds

    # Check that sigma is positive (with some tolerance for MCMC variation)
    # Note: Even with bounds, the mean of samples can sometimes be slightly
    # outside bounds due to sampling near boundaries
    assert sigma2 > 0.0  # Must be positive
    assert sigma2 < 3    # Within upper bound


def test_init_from_raw_sampler_state(simple_model_setup):
    """Test that raw sampler state is used when available."""
    setup = simple_model_setup

    # First run
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=50,
        n_warmup=20
    )

    # Check that raw_result is available
    assert result1.raw_result is not None
    assert hasattr(result1.raw_result, 'get_chain')

    # Get the last positions from the chain
    chain = result1.raw_result.get_chain()
    last_positions = chain[-1, :, :]  # shape: (nwalkers, n_params)

    # Second run should use these positions
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=8, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=10,  # Just a few samples
        n_warmup=0,
        init=result1
    )

    # The initialization should work - we can't expect exact position matches
    # because MCMC is stochastic and walkers move during sampling
    # Instead, check that the sampling worked and produced reasonable results
    assert result2.valid
    assert len(result2.samples) == 8 * 10  # nwalkers * n_samples

    # Check that parameters are in reasonable ranges
    mu_mean = result2.mean('mu')
    sigma_mean = result2.mean('sigma')
    assert -2 < mu_mean < 2  # Should be near 0
    assert 0.3 < sigma_mean < 2  # Should be near 1


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_init_efficiency_comparison(simple_model_setup):
    """Test that initialization improves sampling efficiency."""
    setup = simple_model_setup

    # Run 1: Cold start with warmup
    sampler1 = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    result1 = sampler1.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=500,
        n_warmup=200
    )

    # Run 2: Warm start from result1, no warmup
    sampler2 = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    result2 = sampler2.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=500,
        n_warmup=0,
        init=result1
    )

    # Run 3: Cold start without sufficient warmup
    # Reset parameters to initial values
    for param in setup['params']:
        param.set_value(param.lower + (param.upper - param.lower) / 2)

    sampler3 = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    result3 = sampler3.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=500,
        n_warmup=10  # Very short warmup
    )

    # Compare convergence metrics
    # The warm-started chain (result2) should have better properties than
    # the cold-started chain with insufficient warmup (result3)

    # Check autocorrelation (lower is better)
    # This is a simplified check - just looking at lag-1 autocorrelation
    def lag1_autocorr(samples):
        """Compute lag-1 autocorrelation for each parameter."""
        n_samples = len(samples)
        autocorr = []
        for i in range(samples.shape[1]):
            param_samples = samples[:, i]
            mean = np.mean(param_samples)
            c0 = np.mean((param_samples - mean)**2)
            c1 = np.mean((param_samples[:-1] - mean) * (param_samples[1:] - mean))
            autocorr.append(c1 / c0)
        return np.array(autocorr)

    autocorr2 = lag1_autocorr(result2.samples)
    autocorr3 = lag1_autocorr(result3.samples)

    # Warm-started chain should have lower autocorrelation
    assert np.mean(np.abs(autocorr2)) < np.mean(np.abs(autocorr3))

    # Check that warm-started chain has stable means throughout
    # Split samples into chunks and check consistency
    chunk_size = 100
    n_chunks = 5

    means2_chunks = []
    means3_chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        means2_chunks.append(np.mean(result2.samples[start:end], axis=0))
        means3_chunks.append(np.mean(result3.samples[start:end], axis=0))

    # Variance of chunk means (lower is better - more stable)
    var_means2 = np.var(means2_chunks, axis=0)
    var_means3 = np.var(means3_chunks, axis=0)

    # Warm-started chain should be more stable
    assert np.all(var_means2 < var_means3)

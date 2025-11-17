"""Tests for improved emcee warmup with state continuation."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit._mcmc.emcee import EmceeSampler
from zfit.core.loss import UnbinnedNLL


@pytest.fixture
def simple_model():
    """Create a simple Gaussian model for testing."""
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 0.0, -5, 5, prior=zfit.prior.Normal(0.0, 1.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=zfit.prior.HalfNormal(sigma=1.0))
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Generate data
    data = np.random.normal(0.0, 1.0, 500)
    dataset = zfit.Data.from_numpy(obs=obs, array=data)

    # Create loss
    loss = UnbinnedNLL(model=model, data=dataset)

    return loss, [mu, sigma]


def test_emcee_state_continuation(simple_model):
    """Test that emcee state continuation works correctly."""
    loss, params = simple_model

    # First run
    sampler1 = EmceeSampler(nwalkers=8, verbosity=7)
    result1 = sampler1.sample(loss=loss, params=params, n_samples=50, n_warmup=30)

    # Check that state was stored
    assert result1.info is not None
    assert result1.info.get("type") == "emcee"
    assert result1.info.get("state") is not None

    # Continue from previous run
    sampler2 = EmceeSampler(nwalkers=8, verbosity=7)
    result2 = sampler2.sample(loss=loss, params=params, init=result1, n_samples=50, n_warmup=30)

    # Results should be valid
    assert result2.valid
    assert len(result2.samples) == 8 * 50  # nwalkers * n_samples


def test_emcee_state_continuation_skip_warmup(simple_model):
    """Test that warmup can be skipped when continuing from emcee state."""
    loss, params = simple_model

    # First run with warmup
    sampler1 = EmceeSampler(nwalkers=8)
    result1 = sampler1.sample(loss=loss, params=params, n_samples=50, n_warmup=50)

    # Continue without warmup
    sampler2 = EmceeSampler(nwalkers=8)
    result2 = sampler2.sample(loss=loss, params=params, init=result1, n_samples=100, n_warmup=0)

    # Should work without issues
    assert result2.valid
    assert len(result2.samples) == 8 * 100


def test_emcee_different_nwalkers(simple_model):
    """Test adaptation when number of walkers changes."""
    loss, params = simple_model

    # First run with 8 walkers
    sampler1 = EmceeSampler(nwalkers=8)
    result1 = sampler1.sample(loss=loss, params=params, n_samples=50, n_warmup=30)

    # Continue with more walkers
    sampler2 = EmceeSampler(nwalkers=12)
    result2 = sampler2.sample(loss=loss, params=params, init=result1, n_samples=50, n_warmup=20)

    assert result2.valid
    assert len(result2.samples) == 12 * 50

    # Continue with fewer walkers
    sampler3 = EmceeSampler(nwalkers=6)
    result3 = sampler3.sample(loss=loss, params=params, init=result2, n_samples=50, n_warmup=20)

    assert result3.valid
    assert len(result3.samples) == 6 * 50


def test_non_emcee_init_compatibility(simple_model):
    """Test that non-emcee initialization still works."""
    loss, params = simple_model

    # Create a mock posterior that's not from emcee
    from zfit.mcmc import PosteriorSamples

    # Generate fake samples
    n_fake_samples = 100
    fake_samples = np.random.normal(0, 1, (n_fake_samples, 2))
    fake_samples[:, 0] += 0.1  # Shift mu slightly
    fake_samples[:, 1] = np.abs(fake_samples[:, 1]) + 0.5  # Make sigma positive

    non_emcee_result = PosteriorSamples(
        samples=fake_samples,
        params=params,
        loss=loss,
        sampler=None,
        n_warmup=0,
        n_samples=n_fake_samples,
        info={"type": "other_sampler"}  # Not emcee
    )

    # Should still work, just without state optimization
    sampler = EmceeSampler(nwalkers=8)
    result = sampler.sample(loss=loss, params=params, init=non_emcee_result, n_samples=50, n_warmup=30)

    assert result.valid
    assert len(result.samples) == 8 * 50


def test_parameter_reordering_with_state(simple_model):
    """Test that parameter reordering works with state continuation."""
    loss, params = simple_model

    # First run
    sampler1 = EmceeSampler(nwalkers=8)
    result1 = sampler1.sample(loss=loss, params=params, n_samples=50, n_warmup=30)

    # Continue with parameters in different order
    reversed_params = params[::-1]  # Reverse order
    sampler2 = EmceeSampler(nwalkers=8)
    result2 = sampler2.sample(loss=loss, params=reversed_params, init=result1, n_samples=50, n_warmup=20)

    assert result2.valid
    # Check that parameters are in the expected order
    assert result2.param_names == [p.name for p in reversed_params]


def test_warmup_state_logging(simple_model, capsys):
    """Test that state continuation is properly logged."""
    loss, params = simple_model

    # First run
    sampler1 = EmceeSampler(nwalkers=8, verbosity=7)
    result1 = sampler1.sample(loss=loss, params=params, n_samples=30, n_warmup=20)

    # Clear captured output
    capsys.readouterr()

    # Continue with warmup - should see state continuation message
    sampler2 = EmceeSampler(nwalkers=8, verbosity=7)
    result2 = sampler2.sample(loss=loss, params=params, init=result1, n_samples=30, n_warmup=30)

    captured = capsys.readouterr()
    # Should mention using previous emcee state
    assert "emcee state" in captured.out.lower() or "previous emcee state" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

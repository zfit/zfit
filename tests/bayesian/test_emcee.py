"""Modern pytest tests for EmceeSampler functionality.

Tests EmceeSampler initialization, sampling, and integration with the
new PosteriorSamples API.
"""

#  Copyright (c) 2025 zfit

import pytest
import numpy as np
import zfit
from zfit.loss import UnbinnedNLL


@pytest.fixture
def simple_gaussian_setup():
    """Create a simple Gaussian model setup for testing."""
    # Parameters with priors
    mu = zfit.Parameter("mu", 0.0, prior=zfit.prior.Normal(mu=0.0, sigma=2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10.0,
                          prior=zfit.prior.HalfNormal(sigma=1.0))

    # Generate test data
    zfit.settings.set_seed(42)
    n_points = 500
    data = np.random.normal(0, 1, n_points)
    obs = zfit.Space("x", limits=(-5, 5))
    dataset = zfit.Data.from_numpy(obs=obs, array=data)

    # Create model and loss
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    loss = UnbinnedNLL(model=model, data=dataset)

    return {
        'loss': loss,
        'params': [mu, sigma],
        'mu': mu,
        'sigma': sigma,
        'obs': obs,
        'data': dataset
    }


@pytest.fixture
def uniform_model_setup():
    """Create a uniform model setup for testing parameter limits."""
    # Parameter with limits
    param = zfit.Parameter("test_param", 1.0, 0.0, 2.0,
                          prior=zfit.prior.Uniform(lower=0.0, upper=2.0))

    # Create model and loss
    obs = zfit.Space("x", limits=(-5, 5))
    model = zfit.pdf.Uniform(obs=obs, low=-5, high=param)
    data = zfit.Data.from_numpy(obs=obs, array=np.random.uniform(-5, 5, 100))
    loss = UnbinnedNLL(model=model, data=data)

    return {
        'loss': loss,
        'param': param,
        'obs': obs,
        'data': data
    }


def test_emcee_sampler_default_initialization():
    """Test EmceeSampler default initialization."""
    from zfit._mcmc.emcee import EmceeSampler

    sampler = EmceeSampler()
    assert sampler.nwalkers is None
    assert sampler.name == "EmceeSampler"


@pytest.mark.parametrize("nwalkers", [10, 50])
@pytest.mark.parametrize("verbosity", [0, 7])
def test_emcee_sampler_custom_initialization(nwalkers, verbosity):
    """Test EmceeSampler custom initialization."""
    from zfit._mcmc.emcee import EmceeSampler

    sampler = EmceeSampler(nwalkers=nwalkers, verbosity=verbosity,
                          name="CustomSampler")
    assert sampler.nwalkers == nwalkers
    assert sampler.name == "CustomSampler"


def test_emcee_sampler_via_zfit_mcmc():
    """Test EmceeSampler creation via zfit.mcmc interface."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)
    assert sampler.nwalkers == 16
    assert hasattr(sampler, 'sample')



def test_emcee_sampler_basic_sampling(simple_gaussian_high_stats_posterior):
    """Test basic sampling functionality with new PosteriorSamples API."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Test PosteriorSamples interface
    assert hasattr(posterior, 'get_samples')
    assert hasattr(posterior, 'mean')
    assert hasattr(posterior, 'symerror')
    assert len(posterior.samples) > 0  # Has samples
    assert len(posterior.param_names) == 2

    # Test parameter-specific sampling
    mu_samples = posterior.get_samples(setup['mu'])
    sigma_samples = posterior.get_samples(setup['sigma'])

    assert len(mu_samples) > 1000  # High statistics
    assert len(sigma_samples) > 1000
    assert np.all(np.isfinite(mu_samples))
    assert np.all(np.isfinite(sigma_samples))



@pytest.mark.parametrize("nwalkers", [8, 32])
def test_emcee_sampler_different_nwalkers(simple_gaussian_high_stats_posterior, nwalkers):
    """Test sampling with different numbers of walkers using quick sampling."""
    setup = simple_gaussian_high_stats_posterior
    sampler = zfit.mcmc.EmceeSampler(nwalkers=nwalkers, verbosity=0)

    # Quick sampling just to test nwalkers functionality
    posterior = sampler.sample(
        loss=setup['loss'],
        params=[setup['mu'], setup['sigma']],
        n_samples=20,  # Minimal sampling for speed
        n_warmup=10
    )

    # Check sample count
    assert len(posterior.samples) == 20 * nwalkers

    # Check all samples are finite
    for param in [setup['mu'], setup['sigma']]:
        samples = posterior.get_samples(param)
        assert np.all(np.isfinite(samples))



def test_emcee_sampler_parameter_recovery(simple_gaussian_high_stats_posterior):
    """Test that parameters are recovered reasonably well using high-stats posterior."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Check parameter recovery using new API
    mu_mean = posterior.mean(setup['mu'])
    sigma_mean = posterior.mean(setup['sigma'])

    # Should recover close to true values (mu=0, sigma=1) with better precision due to high stats
    assert float(mu_mean) == pytest.approx(0.0, abs=0.2)  # mu should be close to 0 (tighter bounds with high stats)
    assert float(sigma_mean) == pytest.approx(1.0, abs=0.2)  # sigma should be close to 1 (tighter bounds)


def test_emcee_sampler_parameter_limits_respected(uniform_model_setup):
    """Test that sampling respects parameter limits."""
    from zfit._mcmc.emcee import EmceeSampler

    setup = uniform_model_setup
    sampler = EmceeSampler(nwalkers=10, verbosity=0)

    posterior = sampler.sample(
        loss=setup['loss'],
        params=[setup['param']],
        n_samples=30,
        n_warmup=10
    )

    # Check that all samples are within bounds using new API
    samples = posterior.get_samples(setup['param'])
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 2.0)



def test_emcee_sampler_prior_constraints(simple_gaussian_high_stats_posterior):
    """Test that priors are respected during sampling using high-stats posterior."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Sigma should always be positive (HalfNormal prior)
    sigma_samples = posterior.get_samples(setup['sigma'])
    assert np.all(sigma_samples > 0)


def test_emcee_sampler_posterior_samples_methods(simple_gaussian_setup):
    """Test all PosteriorSamples methods work."""
    from zfit._mcmc.emcee import EmceeSampler

    setup = simple_gaussian_setup
    sampler = EmceeSampler(nwalkers=20, verbosity=0)

    posterior = sampler.sample(
        loss=setup['loss'],
        params=setup['params'],
        n_samples=50,
        n_warmup=25
    )

    # Test all key methods
    for param in setup['params']:
        # Basic statistics
        mean_val = posterior.mean(param)
        symerr_val = posterior.symerror(param)

        assert np.isfinite(mean_val)
        assert symerr_val > 0

        # Credible intervals
        lower, upper = posterior.credible_interval(param, alpha=0.05)
        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

        # Mean should be within credible interval
        assert lower <= mean_val <= upper



def test_emcee_sampler_covariance_matrix(simple_gaussian_high_stats_posterior):
    """Test covariance matrix calculation using high-stats posterior."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Test covariance matrix
    cov_matrix = posterior.covariance()
    assert cov_matrix.shape == (2, 2)  # 2x2 for 2 parameters
    assert np.all(np.isfinite(cov_matrix))
    assert np.all(np.diag(cov_matrix) > 0)  # Diagonal should be positive

    # High-stats should give more precise covariance estimates
    assert np.all(np.abs(cov_matrix) < 10)  # Reasonable magnitude



def test_emcee_sampler_context_manager(simple_gaussian_high_stats_posterior):
    """Test PosteriorSamples context manager using high-stats posterior."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Store original values
    original_mu = setup['mu'].value()
    original_sigma = setup['sigma'].value()

    # Test context manager
    with posterior:
        # Parameters should be set to posterior means inside context
        context_mu = setup['mu'].value()
        context_sigma = setup['sigma'].value()

    # Parameters should be restored after context
    assert setup['mu'].value() == original_mu
    assert setup['sigma'].value() == original_sigma


def test_emcee_sampler_invalid_loss():
    """Test error handling with invalid loss."""
    from zfit._mcmc.emcee import EmceeSampler

    sampler = EmceeSampler(nwalkers=10, verbosity=0)

    with pytest.raises((TypeError, AttributeError)):
        sampler.sample(loss="not_a_loss", n_samples=100)


def test_emcee_sampler_invalid_parameters(simple_gaussian_setup):
    """Test error handling with invalid parameters."""
    from zfit._mcmc.emcee import EmceeSampler

    setup = simple_gaussian_setup
    sampler = EmceeSampler(nwalkers=10, verbosity=0)

    with pytest.raises((TypeError, ValueError)):
        sampler.sample(
            loss=setup['loss'],
            params="not_params",
            n_samples=100
        )

def test_emcee_sampler_invalid_sample_counts(simple_gaussian_setup):
    """Test error handling with invalid sample counts."""
    from zfit._mcmc.emcee import EmceeSampler

    setup = simple_gaussian_setup
    sampler = EmceeSampler(nwalkers=10, verbosity=0)

    # Test negative sample count
    with pytest.raises(ValueError):
        sampler.sample(
            loss=setup['loss'],
            params=setup['params'],
            n_samples=-10
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

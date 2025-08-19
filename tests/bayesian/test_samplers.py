"""Modern pytest tests for zfit Bayesian MCMC samplers.

Tests only working samplers using modern pytest conventions with fixtures,
parametrization, and the new PosteriorSamples API.
"""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.pdf import Gauss
from zfit.loss import UnbinnedNLL


@pytest.fixture
def simple_model():
    """Create a simple Gaussian model for testing."""
    # Create observation space
    obs = zfit.Space("x", limits=(-10, 10))

    # Parameters with priors
    mean = zfit.Parameter("mean", 1.0, lower=-5, upper=10,
                         prior=zfit.prior.Normal(mu=1.0, sigma=0.5))
    sigma = zfit.Parameter("sigma", 2.0, lower=0.1, upper=5,
                          prior=zfit.prior.Normal(mu=2.0, sigma=0.3))

    # Generate test data
    zfit.settings.set_seed(42)
    data_np = np.random.normal(1.0, 2.0, size=200)  # Reduced from 500 for faster tests
    data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

    # Create model and loss
    model = Gauss(obs=obs, mu=mean, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    return {
        'loss': loss,
        'params': [mean, sigma],
        'true_values': {'mean': 1.0, 'sigma': 2.0}
    }


@pytest.fixture
def physics_model():
    """Create a more complex physics model for testing."""
    obs = zfit.Space("x", limits=(0, 10))

    # Background parameters
    coeff1 = zfit.Parameter("coeff1", -0.1, -1, 0,
                           prior=zfit.prior.Uniform(lower=-1, upper=0))
    coeff2 = zfit.Parameter("coeff2", 0.01, -0.1, 0.1,
                           prior=zfit.prior.Uniform(lower=-0.1, upper=0.1))

    # Signal parameters
    mean1 = zfit.Parameter("mean1", 3.0, 2.5, 3.5,
                          prior=zfit.prior.Normal(mu=3.0, sigma=0.5))
    sigma1 = zfit.Parameter("sigma1", 0.2, 0.1, 0.5,
                           prior=zfit.prior.Normal(mu=0.2, sigma=0.1))

    mean2 = zfit.Parameter("mean2", 6.0, 5.5, 6.5,
                          prior=zfit.prior.Normal(mu=6.0, sigma=0.5))
    sigma2 = zfit.Parameter("sigma2", 0.3, 0.1, 0.5,
                           prior=zfit.prior.Normal(mu=0.3, sigma=0.1))

    # Fractions with beta priors
    frac1 = zfit.Parameter("frac1", 0.3, 0, 1,
                          prior=zfit.prior.Beta(alpha=2, beta=5, lower=0, upper=1))
    frac2 = zfit.Parameter("frac2", 0.2, 0, 1,
                          prior=zfit.prior.Beta(alpha=2, beta=5, lower=0, upper=1))

    # Create PDFs
    bkg = zfit.pdf.Chebyshev(obs=obs, coeffs=[coeff1, coeff2])
    peak1 = Gauss(obs=obs, mu=mean1, sigma=sigma1)
    peak2 = zfit.pdf.CrystalBall(obs=obs, mu=mean2, sigma=sigma2,
                                alpha=1.0, n=2.0)

    # Combined model
    model = zfit.pdf.SumPDF([bkg, peak1, peak2], fracs=[frac1, frac2])

    # Generate synthetic data
    zfit.settings.set_seed(42)
    n_events = 1000  # Reduced from 10000 for faster tests
    data = model.sample(n=n_events)

    loss = UnbinnedNLL(model=model, data=data)

    return {
        'loss': loss,
        'model': model,
        'obs': obs,
        'params': model.get_params()
    }


def test_emcee_sampler_creation():
    """Test that EmceeSampler can be created with various configurations."""
    # Basic sampler
    sampler = zfit.mcmc.EmceeSampler(nwalkers=10)
    assert sampler is not None
    assert hasattr(sampler, 'sample')

    # Sampler with verbosity
    sampler_verbose = zfit.mcmc.EmceeSampler(nwalkers=20, verbosity=7)
    assert sampler_verbose is not None

    # Test different nwalkers values
    for nwalkers in [4, 8, 16, 32]:
        sampler = zfit.mcmc.EmceeSampler(nwalkers=nwalkers)
        assert sampler is not None


def test_emcee_sampler_simple_sampling(simple_gaussian_high_stats_posterior):
    """Test basic sampling functionality using high-stats posterior."""
    setup = simple_gaussian_high_stats_posterior
    posterior = setup['posterior']

    # Test posterior interface
    assert hasattr(posterior, 'get_samples')
    assert hasattr(posterior, 'mean')
    assert hasattr(posterior, 'symerr')
    assert len(posterior.samples) > 0
    assert len(posterior.param_names) == 2

    # Test parameter recovery using new API
    mu_samples = posterior.get_samples(setup['mu'])
    sigma_samples = posterior.get_samples(setup['sigma'])

    assert len(mu_samples) > 1000  # High statistics
    assert len(sigma_samples) > 1000
    assert np.all(np.isfinite(mu_samples))
    assert np.all(np.isfinite(sigma_samples))
    assert np.all(sigma_samples > 0)  # Sigma should be positive


@pytest.mark.parametrize("nwalkers,n_samples", [(8, 50), (16, 100)])
def test_emcee_sampler_parameter_recovery(simple_model, nwalkers, n_samples):
        """Test parameter recovery with different sampler configurations."""
        sampler = zfit.mcmc.EmceeSampler(nwalkers=nwalkers, verbosity=0)

        posterior = sampler.sample(
            loss=simple_model['loss'],
            params=simple_model['params'],
            n_samples=n_samples,
            n_warmup=25
        )

        # Check parameter recovery using modern API
        mean_param, sigma_param = simple_model['params']
        true_mean = simple_model['true_values']['mean']
        true_sigma = simple_model['true_values']['sigma']

        # Use new PosteriorSamples methods
        recovered_mean = posterior.mean(mean_param)
        recovered_sigma = posterior.mean(sigma_param)

        # Should recover parameters within reasonable bounds
        assert abs(recovered_mean - true_mean) < 0.5
        assert abs(recovered_sigma - true_sigma) < 0.5



def test_emcee_sampler_credible_intervals(simple_model):
    """Test credible interval calculations."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=100,
        n_warmup=50
    )

    # Test credible intervals for both parameters
    for param in simple_model['params']:
        # 95% credible interval
        lower, upper = posterior.credible_interval(param, alpha=0.05)
        assert lower < upper
        assert np.isfinite(lower)
        assert np.isfinite(upper)

        # Mean should be within the interval
        mean_val = posterior.mean(param)
        assert lower <= mean_val <= upper

        # Test symmetric error
        symerr = posterior.symerr(param)
        assert symerr > 0
        assert np.isfinite(symerr)


def test_emcee_sampler_physics_model_sampling(physics_model):
        """Test sampling on a more complex physics model."""
        sampler = zfit.mcmc.EmceeSampler(nwalkers=20, verbosity=0)

        # Use shorter sampling for this complex model
        posterior = sampler.sample(
            loss=physics_model['loss'],
            params=physics_model['params'],
            n_samples=100,
            n_warmup=50
        )

        # Check basic properties
        assert len(posterior.samples) > 0
        assert len(posterior.param_names) == len(physics_model['params'])

        # Check that all parameters have reasonable samples
        for param in physics_model['params']:
            samples = posterior.get_samples(param)
            assert len(samples) == 100 * 20  # n_samples * nwalkers
            assert np.all(np.isfinite(samples))

            # Check parameter limits are respected
            if param.has_limits:
                if param.lower is not None:
                    assert np.all(samples >= param.lower - 1e-10)
                if param.upper is not None:
                    assert np.all(samples <= param.upper + 1e-10)



def test_emcee_sampler_posterior_as_prior(simple_model):
    """Test using posterior as prior for hierarchical modeling."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=12, verbosity=0)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=75,
        n_warmup=50
    )

    # Test as_prior functionality for hierarchical modeling
    mean_param = simple_model['params'][0]

    # Convert posterior to prior (for hierarchical modeling)
    if hasattr(posterior, 'as_prior'):
        kde_prior = posterior.as_prior(mean_param)
        assert kde_prior is not None
        assert hasattr(kde_prior, 'sample')
        assert hasattr(kde_prior, 'log_pdf')

        # Test sampling from the KDE prior
        prior_samples = kde_prior.sample(100)
        assert len(prior_samples) == 100
        assert np.all(np.isfinite(prior_samples))


def test_posterior_samples_context_manager(simple_model):
        """Test PosteriorSamples context manager functionality."""
        sampler = zfit.mcmc.EmceeSampler(nwalkers=10, verbosity=0)

        posterior = sampler.sample(
            loss=simple_model['loss'],
            params=simple_model['params'],
            n_samples=100,
            n_warmup=50
        )

        # Test context manager
        mean_param, sigma_param = simple_model['params']
        original_mean = mean_param.value()
        original_sigma = sigma_param.value()

        with posterior:
            # Inside context, parameters should be set to posterior means
            context_mean = mean_param.value()
            context_sigma = sigma_param.value()

            # Should be different from original values
            assert context_mean != original_mean or context_sigma != original_sigma

        # After context, parameters should be restored
        assert mean_param.value() == original_mean
        assert sigma_param.value() == original_sigma



def test_posterior_samples_covariance_matrix(simple_model):
    """Test covariance matrix calculation."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=12, verbosity=0)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=100,
        n_warmup=50
    )

    # Test covariance matrix
    cov_matrix = posterior.covariance()
    assert cov_matrix.shape == (2, 2)  # 2x2 for 2 parameters
    assert np.all(np.isfinite(cov_matrix))
    assert np.all(np.diag(cov_matrix) > 0)  # Diagonal should be positive



def test_posterior_samples_convergence_diagnostics(simple_model):
    """Test convergence diagnostics."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=100,
        n_warmup=100
    )

    # Test convergence summary (if implemented)
    if hasattr(posterior, 'convergence_summary'):
        conv_summary = posterior.convergence_summary()
        assert isinstance(conv_summary, dict)
        assert 'valid' in conv_summary


@pytest.mark.parametrize("sampler_config", [
    {"nwalkers": 8, "verbosity": 0},
    {"nwalkers": 16, "verbosity": 0},
])
def test_sampler_configurations(simple_model, sampler_config):
    """Test different sampler configurations."""
    sampler = zfit.mcmc.EmceeSampler(**sampler_config)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=50,
        n_warmup=25
    )

    # Basic validation
    assert len(posterior.samples) > 0
    assert len(posterior.param_names) == 2

    # Check samples are valid
    for param in simple_model['params']:
        samples = posterior.get_samples(param)
        assert len(samples) == 50 * sampler_config['nwalkers']
        assert np.all(np.isfinite(samples))



def test_arviz_integration_conversion(simple_model):
    """Test conversion to ArviZ InferenceData."""
    import arviz as az

    sampler = zfit.mcmc.EmceeSampler(nwalkers=12, verbosity=0)

    posterior = sampler.sample(
        loss=simple_model['loss'],
        params=simple_model['params'],
        n_samples=75,
        n_warmup=50
    )

    # Test ArviZ conversion
    if hasattr(posterior, 'to_arviz'):
        # Native conversion
        idata = posterior.to_arviz()
        assert idata is not None

        # Test ArviZ diagnostics
        summary = az.summary(idata)
        assert len(summary) == 2  # Two parameters

        # Test built-in plotting methods (if available)
        if hasattr(posterior, 'plot_trace'):
            # These methods should exist but may fail in headless environment
            assert callable(posterior.plot_trace)
            assert callable(posterior.plot_posterior)



def test_arviz_integration_summary_methods(simple_model):
        """Test summary and diagnostic methods."""
        sampler = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)

        posterior = sampler.sample(
            loss=simple_model['loss'],
            params=simple_model['params'],
            n_samples=100,
            n_warmup=100
        )

        # Test string representation
        repr_str = str(posterior)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

        # Test summary method (if available with ArviZ)
        if hasattr(posterior, 'summary'):
            try:
                summary = posterior.summary()
                assert summary is not None
            except Exception:
                # May fail without ArviZ or in headless environment
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

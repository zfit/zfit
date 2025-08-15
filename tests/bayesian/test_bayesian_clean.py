"""
Pure pytest-style tests for cleaned-up Bayesian inference functionality.

This test suite validates:
1. All prior distributions work correctly
2. EmceeSampler functionality
3. PosteriorSamples interface
4. Integration with ArviZ
"""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import pytest

import zfit
import arviz as az


# =====================================================================
# Prior distribution tests
# =====================================================================

@pytest.mark.parametrize("prior_name,prior_class,init_params,expected_mean,expected_bounds", [
    ("Normal", zfit.prior.Normal, {"mu": 0.0, "sigma": 1.0}, 0.0, (None, None)),
    ("Uniform", zfit.prior.Uniform, {"lower": 0.0, "upper": 2.0}, 1.0, (0.0, 2.0)),
    ("HalfNormal", zfit.prior.HalfNormal, {"sigma": 1.0}, None, (0.0, None)),
    ("Gamma", zfit.prior.Gamma, {"alpha": 2.0, "beta": 1.0}, None, (0.0, None)),
    ("Beta", zfit.prior.Beta, {"alpha": 2.0, "beta": 2.0}, None, (0.0, 1.0)),
    ("LogNormal", zfit.prior.LogNormal, {"mu": 0.0, "sigma": 1.0}, None, (0.0, None)),
    ("Cauchy", zfit.prior.Cauchy, {"m": 0.0, "gamma": 1.0}, None, (None, None)),
    ("Poisson", zfit.prior.Poisson, {"lam": 3.0}, None, (0.0, None)),
    ("Exponential", zfit.prior.Exponential, {"lam": 2.0}, None, (0.0, None)),  # Changed from rate to lam
    ("StudentT", zfit.prior.StudentT, {"ndof": 5, "mu": 0.0, "sigma": 1.0}, None, (None, None)),
])
def test_prior_distributions_basic_properties(prior_name, prior_class, init_params,
                                            expected_mean, expected_bounds):
    """Test all prior distributions with parametrization."""
    # Create prior
    prior = prior_class(**init_params)

    # Test basic properties
    assert prior is not None
    assert hasattr(prior, 'sample')
    assert hasattr(prior, 'log_pdf')

    # Test sampling
    samples = prior.sample(n=100)
    samples_np = np.array(samples).flatten()
    assert len(samples_np) == 100
    assert np.all(np.isfinite(samples_np))

    # Test bounds
    lower_bound, upper_bound = expected_bounds
    if lower_bound is not None:
        assert np.all(samples_np >= lower_bound - 1e-10), f"{prior_name} violated lower bound"
    if upper_bound is not None:
        assert np.all(samples_np <= upper_bound + 1e-10), f"{prior_name} violated upper bound"

    # Test expected mean (for distributions where we know it)
    if expected_mean is not None:
        assert np.mean(samples_np) == pytest.approx(expected_mean, abs=0.15)

    # Test variance is positive (skip for Cauchy which can have very large variance)
    if prior_name not in ["Cauchy"]:
        assert np.var(samples_np) > 0


# =====================================================================
# EmceeSampler tests
# =====================================================================

@pytest.mark.parametrize("nwalkers", [8, 16, 24])
@pytest.mark.parametrize("verbosity", [0, 5])
def test_emcee_sampler_can_be_created(nwalkers, verbosity):
    """Test sampler can be created with different configurations."""
    sampler = zfit.mcmc.EmceeSampler(nwalkers=nwalkers, verbosity=verbosity)
    assert sampler is not None
    assert sampler.nwalkers == nwalkers


@pytest.mark.parametrize("config_name", ["quick", "medium"])
def test_emcee_samples_simple_gaussian(build_simple_gaussian, make_sampler,
                                     assert_parameter_recovered, config_name):
    """Test sampling from a simple Gaussian model with different configs."""
    configs = {
        "quick": {"nwalkers": 8, "n_samples": 50, "n_warmup": 20},
        "medium": {"nwalkers": 16, "n_samples": 200, "n_warmup": 100},
    }
    config = configs[config_name]

    # Build model
    model_dict = build_simple_gaussian()

    # Create sampler
    sampler = make_sampler(nwalkers=config["nwalkers"])

    # Sample
    posterior = sampler.sample(
        loss=model_dict['loss'],
        params=model_dict['params'],
        n_samples=config["n_samples"],
        n_warmup=config["n_warmup"]
    )

    # Basic checks
    assert posterior is not None
    assert hasattr(posterior, 'get_samples')

    # Check parameter recovery
    mu_samples = posterior.get_samples(model_dict['mu'])
    sigma_samples = posterior.get_samples(model_dict['sigma'])

    assert len(mu_samples) == config["n_samples"] * config["nwalkers"]
    assert len(sigma_samples) == config["n_samples"] * config["nwalkers"]

    # Should recover parameters within reasonable bounds
    assert_parameter_recovered(mu_samples, 0.0, tolerance=0.5)
    assert_parameter_recovered(sigma_samples, 1.0, tolerance=0.5)


def test_emcee_samples_extended_model(build_simple_gaussian, make_sampler, assert_posterior_valid):
    """Test sampling from an extended model."""
    # Build extended model
    model_dict = build_simple_gaussian(extended=True, n_events=800)

    # Sample
    sampler = make_sampler(nwalkers=16)
    posterior = sampler.sample(
        loss=model_dict['loss'],
        params=model_dict['params'],
        n_samples=150,
        n_warmup=100
    )

    # Validate posterior
    assert_posterior_valid(posterior, expected_params=["mu", "sigma", "n_sig"])

    # Check all parameters can be sampled
    for param in model_dict['params']:
        samples = posterior.get_samples(param)
        assert len(samples) == 150 * 16
        assert np.isfinite(samples).all()


# =====================================================================
# PosteriorSamples interface tests
# =====================================================================

@pytest.fixture
def simple_posterior(build_simple_gaussian, sample_posterior):
    """Create a simple posterior for testing."""
    model_dict = build_simple_gaussian()
    posterior = sample_posterior(
        model_dict['loss'],
        model_dict['params'],
        nwalkers=8,
        n_samples=100,
        n_warmup=50
    )
    return posterior, model_dict


def test_posterior_samples_has_required_interface(simple_posterior):
    """Test PosteriorSamples provides expected interface."""
    posterior, model_dict = simple_posterior
    mu = model_dict['mu']

    # Test interface
    required_methods = ['get_samples', 'mean', 'symerror', 'std',
                       'credible_interval', 'as_prior', 'covariance']

    for method in required_methods:
        assert hasattr(posterior, method), f"Missing method: {method}"

    # Test basic functionality
    samples = posterior.get_samples(mu)
    assert len(samples) == 100 * 8  # 100 samples × 8 walkers
    assert np.isfinite(samples).all()

    # Test statistical methods
    mean_val = posterior.mean(mu)
    error_val = posterior.symerr(mu)
    std_val = posterior.std(mu)
    ci_lower, ci_upper = posterior.credible_interval(mu)

    assert np.isfinite(mean_val)
    assert error_val > 0
    assert std_val > 0
    assert ci_lower < ci_upper


@pytest.mark.parametrize("alpha,sigma", [
    (0.05, None),    # 95% CI using alpha
    (0.32, None),    # ~68% CI using alpha
    (None, 1),       # 1 sigma using sigma
    (None, 2),       # 2 sigma using sigma
])
def test_posterior_credible_intervals(simple_posterior, alpha, sigma):
    """Test credible interval calculation with different parameters."""
    posterior, model_dict = simple_posterior

    # Calculate credible interval
    if sigma is not None:
        ci_lower, ci_upper = posterior.credible_interval(model_dict['params'], sigma=sigma)
    else:
        ci_lower, ci_upper = posterior.credible_interval(model_dict['params'], alpha=alpha)

    # Check properties
    assert len(ci_lower) == len(model_dict['params'])
    assert len(ci_upper) == len(model_dict['params'])
    assert np.all(ci_lower < ci_upper)
    assert np.all(np.isfinite(ci_lower))
    assert np.all(np.isfinite(ci_upper))


def test_posterior_context_manager(simple_posterior):
    """Test context manager functionality for setting parameter values."""
    posterior, model_dict = simple_posterior
    params = model_dict['params']

    # Store original values
    original_values = [p.value() for p in params]

    # Use context manager
    with posterior:
        # Parameters should be set to posterior means
        for i, param in enumerate(params):
            assert param.value() != original_values[i]
            assert float(param.value()) == pytest.approx(float(posterior.mean(param)), abs=1e-6)

    # Parameters should be restored
    for i, param in enumerate(params):
        assert param.value() == original_values[i]


def test_posterior_as_prior(simple_posterior, assert_samples_valid):
    """Test converting posterior to prior for hierarchical modeling."""
    posterior, model_dict = simple_posterior
    mu = model_dict['mu']

    # Create KDE prior from posterior
    kde_prior = posterior.as_prior(mu)

    assert kde_prior is not None
    assert hasattr(kde_prior, 'sample')
    assert hasattr(kde_prior, 'log_pdf')

    # Test sampling from the KDE prior
    kde_samples = kde_prior.sample(100)
    assert_samples_valid(kde_samples, expected_shape=(100, 1))


# =====================================================================
# ArviZ integration tests
# =====================================================================

def test_posterior_to_arviz_conversion(gaussian_posterior_highstats):
    """Test conversion of results to ArviZ format."""
    posterior = gaussian_posterior_highstats['posterior']

    # Convert to ArviZ
    inference_data = posterior.to_arviz()
    assert inference_data is not None

    # Test ArviZ functionality
    summary = az.summary(inference_data)
    assert len(summary) == 2  # Two parameters (mu, sigma)

    # Check diagnostics available
    r_hat = az.rhat(inference_data)
    assert 'mu' in r_hat
    assert 'sigma' in r_hat

    ess = az.ess(inference_data)
    assert 'mu' in ess
    assert 'sigma' in ess


@pytest.mark.parametrize("method", ["bulk", "tail", "mean", "sd"])
def test_arviz_diagnostic_methods(physics_posterior_highstats, method):
    """Test different ArviZ diagnostic methods."""
    posterior = physics_posterior_highstats['posterior']
    inference_data = posterior.to_arviz()

    if method in ["bulk", "tail"]:
        # ESS methods
        result = az.ess(inference_data, method=method)
    else:
        # MCSE methods
        result = az.mcse(inference_data, method=method)

    assert len(result) == len(posterior.param_names)

    # All values should be positive
    for param_name in posterior.param_names:
        assert float(result[param_name]) > 0


# =====================================================================
# Implementation cleanliness tests
# =====================================================================

def test_import_structure_is_clean():
    """Test clean import structure."""
    # Test main imports work
    from zfit import prior
    from zfit import mcmc

    # Test all expected priors are available
    expected_priors = [
        'Normal', 'Uniform', 'HalfNormal', 'Gamma', 'Beta',
        'LogNormal', 'Cauchy', 'Poisson', 'Exponential', 'StudentT'
    ]

    for prior_name in expected_priors:
        assert hasattr(prior, prior_name), f"Missing prior: {prior_name}"

    # Test sampler is available
    assert hasattr(mcmc, 'EmceeSampler')


def test_broken_samplers_not_exposed():
    """Test that broken samplers are not exposed in public API."""
    import zfit.mcmc as mcmc_module

    # These samplers should NOT be available
    broken_samplers = ['NUTSSampler', 'ZeusSampler', 'DynestySampler', 'UltraNestSampler']

    for sampler_name in broken_samplers:
        assert not hasattr(mcmc_module, sampler_name), \
            f"Broken sampler {sampler_name} should not be exposed"


# =====================================================================
# End-to-end workflow tests
# =====================================================================

@pytest.mark.slow
@pytest.mark.sampling
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_end_to_end_bayesian_workflow(build_physics_model, make_sampler,
                                    assert_posterior_valid, assert_parameter_recovered):
    """Test complete end-to-end Bayesian workflow with physics model."""
    # Build model
    model_dict = build_physics_model(n_sig1=500, n_sig2=300, n_bkg=200)

    # Sample
    sampler = make_sampler(nwalkers=32)
    posterior = sampler.sample(
        loss=model_dict['loss'],
        params=model_dict['params'],
        n_samples=300,
        n_warmup=200
    )

    # Validate posterior
    param_names = [p.name for p in model_dict['params']]
    assert_posterior_valid(posterior, expected_params=param_names)

    # Check parameter recovery
    for param in model_dict['params']:
        samples = posterior.get_samples(param)
        mean_val = np.mean(samples)

        # Should be within parameter bounds
        if hasattr(param, 'lower') and param.lower is not None:
            assert mean_val > param.lower
        if hasattr(param, 'upper') and param.upper is not None:
            assert mean_val < param.upper

    # ArviZ integration
    inference_data = posterior.to_arviz()
    summary = az.summary(inference_data)
    assert len(summary) == len(model_dict['params'])

    # Check convergence
    r_hat = az.rhat(inference_data)
    for param_name in param_names:
        assert float(r_hat[param_name]) < 1.1  # Good convergence


# =====================================================================
# Prior-parameter integration tests
# =====================================================================

@pytest.mark.parametrize("prior_type,param_bounds", [
    ("normal", (-5, 5)),      # Bounded parameter with unbounded prior
    ("uniform", (0, 2)),      # Bounded parameter with uniform prior
    ("halfnormal", (0, None)), # Lower-bounded parameter
])
def test_prior_respects_parameter_bounds(make_parameter, all_prior_factories,
                                       assert_samples_valid, prior_type, param_bounds):
    """Test that priors respect parameter bounds."""
    lower, upper = param_bounds

    # Create bounded parameter
    param = make_parameter("test", 1.0, lower=lower, upper=upper)

    # Create prior
    if prior_type == "normal":
        prior = all_prior_factories["normal"](mu=0.0, sigma=10.0)  # Wide prior
    elif prior_type == "uniform":
        prior = all_prior_factories["uniform"]()  # No bounds specified
    elif prior_type == "halfnormal":
        prior = all_prior_factories["halfnormal"](sigma=5.0)

    param.set_prior(prior)

    # Sample from the prior through the parameter
    samples = param.prior.sample(1000)
    validated = assert_samples_valid(samples)

    # Check bounds are respected
    if lower is not None:
        assert np.all(validated >= lower - 1e-10)
    if upper is not None:
        assert np.all(validated <= upper + 1e-10)


# =====================================================================
# Performance and edge case tests
# =====================================================================

@pytest.mark.parametrize("n_params", [1, 5, 10])
def test_emcee_scales_with_parameter_count(make_space, make_parameter,
                                          make_gaussian_pdf, make_loss, make_sampler, n_params):
    """Test that EmceeSampler works with different numbers of parameters."""
    obs = make_space("x", -5, 5)

    # Create multiple parameters with priors
    params = []
    for i in range(n_params):
        # Add a simple normal prior to each parameter
        p = make_parameter(f"p{i}", 0.0, -1, 1)
        p.set_prior(zfit.prior.Normal(0.0, 0.5))
        params.append(p)

    # Simple sum model
    model = make_gaussian_pdf(obs, mu=sum(params), sigma=1.0)
    data = model.sample(100)
    loss = make_loss(model, data)

    # Sample with appropriate number of walkers
    sampler = make_sampler(nwalkers=max(2*n_params + 2, 8))
    posterior = sampler.sample(loss, params, n_samples=50, n_warmup=20)

    assert posterior is not None
    assert len(posterior.param_names) == n_params


@pytest.mark.flaky(reruns=3)
def test_posterior_convergence_diagnostics(gaussian_posterior_highstats):
    """Test convergence diagnostics on high-statistics posterior."""
    posterior = gaussian_posterior_highstats['posterior']

    # Should have converged
    assert posterior.valid
    assert posterior.converged

    # Check R-hat values
    if posterior.rhat is not None:
        assert np.all(posterior.rhat < 1.1)

    # Check ESS values
    if posterior.ess is not None:
        assert np.all(posterior.ess > 100)

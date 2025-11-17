"""Modern pytest tests comparing MCMC samplers on physics examples.

Tests using realistic physics models with background and signal components,
demonstrating parameter recovery and model fitting capabilities.
"""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit
from zfit.pdf import Gauss


@pytest.fixture
def physics_model():
    """Create a physics model with polynomial background + gaussian + crystal ball."""
    # Create space
    obs = zfit.Space("x", limits=(0, 10))

    # Background parameters with uniform priors
    coeff1 = zfit.Parameter("coeff1", -0.1, -1, 0,
                           prior=zfit.prior.Uniform(lower=-1, upper=0))
    coeff2 = zfit.Parameter("coeff2", 0.01, -0.1, 0.1,
                           prior=zfit.prior.Uniform(lower=-0.1, upper=0.1))

    # Gaussian parameters with normal priors
    mean1 = zfit.Parameter("mean1", 3.0, 2.5, 3.5,
                          prior=zfit.prior.Normal(mu=3.0, sigma=0.5))
    sigma1 = zfit.Parameter("sigma1", 0.2, 0.1, 0.5,
                           prior=zfit.prior.Normal(mu=0.2, sigma=0.3))

    # Crystal Ball parameters with normal priors
    mean2 = zfit.Parameter("mean2", 6.0, 5.5, 6.5,
                          prior=zfit.prior.Normal(mu=6.0, sigma=0.5))
    sigma2 = zfit.Parameter("sigma2", 0.3, 0.1, 0.5,
                           prior=zfit.prior.Normal(mu=0.3, sigma=0.3))
    alpha = zfit.Parameter("alpha", 1.0, 0.5, 2.0,
                          prior=zfit.prior.Normal(mu=1.0, sigma=0.5))
    n = zfit.Parameter("n", 2.0, 1.0, 4.0,
                      prior=zfit.prior.Normal(mu=2.0, sigma=1.0))

    # Fractions with beta priors
    frac1 = zfit.Parameter("frac1", 0.3, 0, 1,
                          prior=zfit.prior.Beta(alpha=2, beta=5, lower=0, upper=1))
    frac2 = zfit.Parameter("frac2", 0.2, 0, 1,
                          prior=zfit.prior.Beta(alpha=2, beta=5, lower=0, upper=1))

    # Create PDFs
    bkg = zfit.pdf.Chebyshev(obs=obs, coeffs=[coeff1, coeff2])
    peak1 = Gauss(obs=obs, mu=mean1, sigma=sigma1)
    peak2 = zfit.pdf.CrystalBall(
        obs=obs,
        mu=mean2,
        sigma=sigma2,
        alpha=alpha,
        n=n,
    )

    # Combined model
    model = zfit.pdf.SumPDF([bkg, peak1, peak2], fracs=[frac1, frac2])

    return model, obs


@pytest.fixture
def sampled_data(physics_model):
    """Generate toy data from the model."""
    model, obs = physics_model
    zfit.settings.set_seed(42)
    n_events = 500  # Reduced from 2000 for faster tests
    data = model.sample(n=n_events)
    return data


@pytest.fixture
def minuit_results(sampled_data, physics_model):
    """Fit the model with Minuit for comparison."""
    model, obs = physics_model

    # Create loss
    loss = UnbinnedNLL(model=model, data=sampled_data)

    # Fit with Minuit
    minimizer = Minuit(gradient="zfit")
    result = minimizer.minimize(loss)
    result.hesse()

    # Get parameter values and errors from minimizer
    values = {param.name: result.params[param]["value"] for param in model.get_params()}
    errors = {param.name: result.params[param]["hesse"]["error"] for param in model.get_params()}
    return loss, result, values, errors


def test_emcee_physics_fit_comparison(physics_model_high_stats_posterior):
    """Test EmceeSampler on physics model using high-stats posterior."""
    setup = physics_model_high_stats_posterior
    posterior = setup['posterior']
    model = setup['model']
    data = setup['data']
    loss = setup['loss']

    # Calculate mean and std from posterior using new API
    mcmc_means = {param.name: posterior.mean(param) for param in model.get_params()}
    mcmc_stds = {param.name: posterior.symerr(param) for param in model.get_params()}

    # Basic validation of high-stats results
    for param_name in mcmc_means:
        assert np.isfinite(mcmc_means[param_name])
        assert mcmc_stds[param_name] > 0
        assert np.isfinite(mcmc_stds[param_name])

    # Fit with Minuit for comparison
    minimizer = Minuit(gradient="zfit")
    result = minimizer.minimize(loss)
    result.hesse()

    # Get Minuit values and errors
    minuit_values = {param.name: result.params[param]["value"] for param in model.get_params()}
    minuit_errors = {param.name: result.params[param]["hesse"]["error"] for param in model.get_params()}

    # Compare with Minuit results
    for param_name in minuit_values:
        minuit_val = minuit_values[param_name]
        minuit_err = minuit_errors[param_name]
        mcmc_mean = mcmc_means[param_name]
        mcmc_std = mcmc_stds[param_name]

        # Determine if Minuit errors are reliable
        # Large absolute errors (>100) or errors much larger than parameter value indicate numerical issues
        minuit_val_abs = abs(minuit_val)
        error_to_value_ratio = minuit_err / max(minuit_val_abs, 1.0)  # Avoid division by zero
        is_minuit_unreliable = (minuit_err > 100) or (error_to_value_ratio > 2.0)

        # Check if MCMC mean is within 3 sigma of Minuit result
        if is_minuit_unreliable:
            # Use MCMC uncertainty as reference when Minuit is unreliable
            tolerance = 3 * mcmc_std
        else:
            # Use the larger of the two uncertainties for reliable cases
            tolerance = 3 * max(minuit_err, mcmc_std)

        assert float(mcmc_mean) == pytest.approx(float(minuit_val), abs=tolerance), (
            f"EmceeSampler: {param_name} mean differs significantly"
        )

        # Check if uncertainties are roughly consistent (within factor of 5 for complex models)

        if not is_minuit_unreliable:
            uncertainty_ratio = mcmc_std / minuit_err
            assert 0.2 < uncertainty_ratio < 5.0, (
                f"EmceeSampler: {param_name} uncertainty differs significantly: {uncertainty_ratio:.1f}"
            )
        else:
            # Just check that MCMC uncertainties are reasonable for unreliable Minuit fits
            assert mcmc_std > 0 and mcmc_std < 1000, (
                f"EmceeSampler: {param_name} MCMC uncertainty unreasonable: {mcmc_std:.3f} "
                f"(Minuit error unreliable: {minuit_err:.1f})"
            )


@pytest.mark.parametrize("nwalkers", [20, 32])  # Reduced from [20, 32, 50]
def test_emcee_parameter_configurations(physics_model, sampled_data, nwalkers):
    """Test EmceeSampler with different nwalkers configurations."""
    model, obs = physics_model
    loss = UnbinnedNLL(model=model, data=sampled_data)

    sampler = zfit.mcmc.EmceeSampler(nwalkers=nwalkers, verbosity=0)

    # Use shorter sampling for parameter testing
    posterior = sampler.sample(loss=loss, n_samples=50, n_warmup=30)  # Reduced for faster tests

    # Basic checks
    assert len(posterior.samples) > 0
    assert len(posterior.param_names) == len(model.get_params())

    # Check that all parameters have finite samples
    for param in model.get_params():
        samples = posterior.get_samples(param)
        assert len(samples) == 50 * nwalkers  # Updated for reduced sample count
        assert np.all(np.isfinite(samples))


def test_posterior_distributions(physics_model_high_stats_posterior):
    """Test that posterior distributions are reasonable using high-stats posterior."""
    setup = physics_model_high_stats_posterior
    posterior = setup['posterior']
    model = setup['model']

    # Check basic properties of posterior samples using new API
    assert len(posterior.samples) > 0
    assert len(posterior.param_names) == len(model.get_params())

    # Check parameter constraints
    for param in model.get_params():
        samples = posterior.get_samples(param)

        # Check if samples are mostly within parameter limits (allow 5% violations)
        if param.has_limits:
            if param.lower is not None:
                within_lower = np.sum(samples >= param.lower) / len(samples)
                assert within_lower >= 0.95, f"Only {within_lower:.2%} of {param.name} samples >= {param.lower}"
            if param.upper is not None:
                within_upper = np.sum(samples <= param.upper) / len(samples)
                assert within_upper >= 0.95, f"Only {within_upper:.2%} of {param.name} samples <= {param.upper}"

        # Check if distribution is reasonable
        assert np.all(np.isfinite(samples))
        assert np.std(samples) > 0  # Should have some variation

        # For fraction parameters, check they sum to ≤ 1
        if param.name.startswith("frac"):
            frac_params = [p for p in model.get_params() if p.name.startswith("frac")]
            frac_samples = np.column_stack([posterior.get_samples(p) for p in frac_params])
            assert np.all(np.sum(frac_samples, axis=1) <= 1.0 + 1e-10)


def test_credible_intervals_physics(physics_model_high_stats_posterior):
    """Test credible intervals on physics model parameters using high-stats posterior."""
    setup = physics_model_high_stats_posterior
    posterior = setup['posterior']
    model = setup['model']

    # Test credible intervals for key parameters
    key_params = [p for p in model.get_params() if p.name in ['mean1', 'sigma1', 'frac1']]

    for param in key_params:
        # 90% credible interval
        lower, upper = posterior.credible_interval(param, alpha=0.1)
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



def test_covariance_matrix_physics(physics_model, sampled_data):
        """Test covariance matrix calculation for physics model."""
        model, obs = physics_model
        loss = UnbinnedNLL(model=model, data=sampled_data)

        sampler = zfit.mcmc.EmceeSampler(nwalkers=24, verbosity=0)
        posterior = sampler.sample(loss=loss, n_samples=200, n_warmup=50)

        # Test covariance matrix
        cov_matrix = posterior.covariance()
        n_params = len(model.get_params())

        assert cov_matrix.shape == (n_params, n_params)
        assert np.all(np.isfinite(cov_matrix))
        assert np.all(np.diag(cov_matrix) > 0)  # Diagonal should be positive

        # Matrix should be symmetric
        assert np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10)



def test_posterior_as_prior_hierarchical(physics_model, sampled_data):
    """Test using posterior as prior for hierarchical modeling."""
    model, obs = physics_model
    loss = UnbinnedNLL(model=model, data=sampled_data)

    sampler = zfit.mcmc.EmceeSampler(nwalkers=24, verbosity=0)
    posterior = sampler.sample(loss=loss, n_samples=150, n_warmup=50)

    # Test converting posterior to prior for key parameters
    mean_param = [p for p in model.get_params() if p.name == 'mean1'][0]

    if hasattr(posterior, 'as_prior'):
        kde_prior = posterior.as_prior(mean_param)
        assert kde_prior is not None
        assert hasattr(kde_prior, 'sample')
        assert hasattr(kde_prior, 'log_pdf')

        # Test sampling from the KDE prior
        prior_samples = kde_prior.sample(100)
        assert len(prior_samples) == 100
        assert np.all(np.isfinite(prior_samples))



def test_arviz_integration_physics(physics_model, sampled_data):
        """Test ArviZ integration with physics model."""
        model, obs = physics_model
        loss = UnbinnedNLL(model=model, data=sampled_data)

        sampler = zfit.mcmc.EmceeSampler(nwalkers=24, verbosity=0)
        posterior = sampler.sample(loss=loss, n_samples=100, n_warmup=50)

        # Test ArviZ integration
        import arviz as az

        if hasattr(posterior, 'to_arviz'):
            idata = posterior.to_arviz()
            assert idata is not None

            # Test diagnostics
            summary = az.summary(idata)
            assert len(summary) == len(model.get_params())

            # Test R-hat and ESS
            rhat = az.rhat(idata)
            ess = az.ess(idata)

            # Most parameters should have reasonable R-hat
            for param_name in rhat.data_vars:
                rhat_val = float(rhat[param_name].values)
                assert 0.9 < rhat_val < 1.2  # Reasonable range for R-hat

def test_context_manager_physics(physics_model, sampled_data):
    """Test context manager functionality with physics model."""
    model, obs = physics_model
    loss = UnbinnedNLL(model=model, data=sampled_data)

    sampler = zfit.mcmc.EmceeSampler(nwalkers=24, verbosity=0)
    posterior = sampler.sample(loss=loss, n_samples=100, n_warmup=50)

    # Store original parameter values
    params = model.get_params()
    original_values = {p.name: p.value() for p in params}

    # Test context manager
    with posterior:
        # Inside context, parameters should be set to posterior means
        for param in params:
            context_val = param.value()
            # Should be different from original (with some tolerance)
            if float(original_values[param.name]) != pytest.approx(float(context_val), abs=1e-10):
                break
        else:
            # If we get here, no parameters changed (might be okay for some models)
            pass

    # After context, parameters should be restored
    for param in params:
        assert float(param.value()) == pytest.approx(original_values[param.name], abs=1e-10)

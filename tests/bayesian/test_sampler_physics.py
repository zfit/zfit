"""Tests comparing different MCMC samplers on a physics example."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import zfit
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit
from zfit.pdf import Gauss, GeneralizedCB


@pytest.fixture
def physics_model():
    """Create a physics model with polynomial background + gaussian + gen CB."""
    # Create space
    obs = zfit.Space("x", limits=(0, 10))

    # Background parameters with uniform priors
    coeff1 = zfit.Parameter("coeff1", -0.1, -1, 0, prior=zfit.prior.UniformPrior(-1, 0))
    coeff2 = zfit.Parameter("coeff2", 0.01, -0.1, 0.1, prior=zfit.prior.UniformPrior(-0.1, 0.1))

    # Gaussian parameters with normal priors
    mean1 = zfit.Parameter("mean1", 3.0, 2.5, 3.5, prior=zfit.prior.NormalPrior(3.0, 0.5))
    sigma1 = zfit.Parameter("sigma1", 0.2, 0.1, 0.5, prior=zfit.prior.NormalPrior(0.2, 0.3))

    # Generalized CB parameters with normal priors
    mean2 = zfit.Parameter("mean2", 6.0, 5.5, 6.5, prior=zfit.prior.NormalPrior(6.0, 0.5))
    sigma2 = zfit.Parameter("sigma2", 0.3, 0.1, 0.5, prior=zfit.prior.NormalPrior(0.3, 0.3))
    alpha = zfit.Parameter("alpha", 1.0, 0.5, 2.0, prior=zfit.prior.NormalPrior(1.0, 0.5))
    n = zfit.Parameter("n", 2.0, 1.0, 4.0, prior=zfit.prior.NormalPrior(2.0, 1.0))

    # Fractions with beta priors
    frac1 = zfit.Parameter("frac1", 0.3, 0, 1, prior=zfit.prior.BetaPrior(alpha=2, beta=5))
    frac2 = zfit.Parameter("frac2", 0.2, 0, 1, prior=zfit.prior.BetaPrior(alpha=2, beta=5))

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
    np.random.seed(42)
    n_events = 10000
    data = model.sample(n=n_events)
    return data


@pytest.fixture
def minuitresults(sampled_data, physics_model):
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


@pytest.mark.parametrize("sampler_name", ["emcee", "nuts", "pt", "smc", "ultranest", "stan"])
def test_physics_fit_comparison(sampler_name, minuitresults):
    """Compare results from minimizer and different MCMC samplers."""

    # List of samplers to test
    verbosity = 7
    samplers = {
        "emcee": zfit.mcmc.EmceeSampler(nwalkers=50, verbosity=verbosity),
        # "nuts": zfit.mcmc.NUTSSampler(step_size=0.1, verbosity=verbosity),
        # "pt": zfit.mcmc.PTSampler(nwalkers=20, ntemps=3, verbosity=verbosity),
        # "smc": zfit.mcmc.SMCSampler(n_particles=100, verbosity=verbosity),
        # "ultranest": zfit.mcmc.UltraNestSampler(min_num_live_points=400, verbosity=verbosity),
        # "stan": zfit.mcmc.CustomStanSampler(verbosity=verbosity),
    }

    sampler = samplers.get(sampler_name)
    if sampler is None:
        pytest.skip(f"Sampler {sampler_name} not implemented or not available.")
    print(f"\nTesting {sampler_name} sampler...")
    loss, result, minuit_values, minuit_errors = minuitresults
    model = loss.model[0]

    # Sample from posterior
    posterior = sampler.sample(loss=loss, n_samples=1000, n_warmup=200)

    # Calculate mean and std from posterior
    mcmc_means = {param.name: np.mean(posterior.samples[:, i]) for i, param in enumerate(model.get_params())}
    mcmc_stds = {param.name: np.std(posterior.samples[:, i]) for i, param in enumerate(model.get_params())}

    # Compare with Minuit results
    for param_name in minuit_values:
        minuit_val = minuit_values[param_name]
        minuit_err = minuit_errors[param_name]
        mcmc_mean = mcmc_means[param_name]
        mcmc_std = mcmc_stds[param_name]

        # Check if MCMC mean is within 3 sigma of Minuit result
        assert abs(mcmc_mean - minuit_val) < 3 * max(minuit_err, mcmc_std), (
            f"{sampler_name}: {param_name} mean differs significantly"
        )

        # Check if uncertainties are roughly consistent
        assert 0.5 < mcmc_std / minuit_err < 2.0, f"{sampler_name}: {param_name} uncertainty differs significantly"

    print(f"{sampler_name} sampler results consistent with Minuit")


def test_posterior_distributions(physics_model, sampled_data):
    """Test that posterior distributions are reasonable."""
    model, obs = physics_model
    loss = UnbinnedNLL(model=model, data=sampled_data)

    # Use NUTS sampler for this test
    sampler = zfit._mcmc.EmceeSampler(step_size=0.1)

    try:
        posterior = sampler.sample(loss=loss, n_samples=1000, n_warmup=200)

        # Check basic properties of posterior samples
        assert posterior.samples.shape[1] == len(model.get_params())

        # Check parameter constraints
        for i, param in enumerate(model.get_params()):
            samples = posterior.samples[:, i]

            # Check if samples are within parameter limits
            if param.has_limits:
                if param.lower is not None:
                    assert np.all(samples >= param.lower)
                if param.upper is not None:
                    assert np.all(samples <= param.upper)

            # Check if distribution is reasonable
            assert np.all(np.isfinite(samples))
            assert np.std(samples) > 0  # Should have some variation

            # For fraction parameters, check they sum to ≤ 1
            if param.name.startswith("frac"):
                fracs = posterior.samples[:, [j for j, p in enumerate(model.get_params()) if p.name.startswith("frac")]]
                assert np.all(np.sum(fracs, axis=1) <= 1.0 + 1e-10)

    except ImportError as e:
        pytest.skip(f"Skipping NUTS sampler test due to: {str(e)}")

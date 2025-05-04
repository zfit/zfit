#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import pytest

import zfit
from zfit._mcmc.emcee import EmceeSampler
from zfit.core.loss import UnbinnedNLL


@pytest.fixture
def obs():
    return zfit.Space("x", limits=(-10, 10))


@pytest.fixture
def true_params():
    return {"mu": 1.5, "sigma": 0.5}


@pytest.fixture
def model_params(true_params):
    mu = zfit.Parameter("mu", true_params["mu"], -5, 5, prior=zfit.prior.NormalPrior(1.5, 1.2))
    sigma = zfit.Parameter("sigma", true_params["sigma"], 0.1, 10, prior=zfit.prior.HalfNormalPrior(mu=1.0, sigma=1.0))
    return {"mu": mu, "sigma": sigma}


@pytest.fixture
def gauss(obs, model_params):
    return zfit.pdf.Gauss(mu=model_params["mu"], sigma=model_params["sigma"], obs=obs)


@pytest.fixture
def data(obs, true_params):
    n_events = 1000
    data = np.random.normal(true_params["mu"], true_params["sigma"], n_events)
    return zfit.Data.from_numpy(obs=obs, array=data)


@pytest.fixture
def nll(gauss, data):
    return UnbinnedNLL(model=gauss, data=data)


def test_posteriors_basic(nll, model_params):
    # Run MCMC
    nwalkers = 5
    sampler = EmceeSampler(nwalkers=nwalkers)
    n_samples = 500
    n_warmup = 100
    posteriors = sampler.sample(loss=nll, n_samples=n_samples, n_warmup=n_warmup)

    # Test basic properties
    assert len(posteriors.param_names) == 2
    assert posteriors.n_samples == n_samples
    assert posteriors.n_warmup == n_warmup
    assert posteriors.samples.shape == (n_samples * nwalkers, 2)  # n_samples * nwalkers, n_params

    # Test parameter recovery
    mu_mean = posteriors.mean("mu")
    sigma_mean = posteriors.mean("sigma")
    assert abs(mu_mean - 1.5) < 0.2
    assert abs(sigma_mean - 0.5) < 0.2

    # Test credible intervals
    mu_lower, mu_upper = posteriors.credible_interval("mu")
    assert mu_lower < 1.5 < mu_upper

    # Test HDI
    mu_hdi_lower, mu_hdi_upper = posteriors.highest_density_interval("mu")
    assert mu_hdi_lower < 1.5 < mu_hdi_upper

    # Test summary
    summary = posteriors.summary()
    assert "mu" in summary
    assert "sigma" in summary
    assert "mean" in summary["mu"]
    assert "std" in summary["mu"]

    # Test parameter bounds are respected
    assert np.all(posteriors.samples[:, 1] > 0.1)  # sigma lower bound
    assert np.all(posteriors.samples[:, 0] > -5)  # mu lower bound
    assert np.all(posteriors.samples[:, 0] < 5)  # mu upper bound


def test_posteriors_error_handling(nll, model_params):
    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    with pytest.raises(ValueError):
        posteriors.mean("nonexistent_param")

    with pytest.raises(IndexError):
        posteriors.mean(123)  # Invalid parameter type


# TODO bayesian: add plotting functionality?
# def test_posteriors_plotting(nll, model_params):
#     pytest.importorskip("matplotlib")
#
#     sampler = EmceeSampler()
#     posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)
#
#     # Test basic plotting functions
#     posteriors.plot_trace("mu")
#     posteriors.plot_posterior("mu")
#     posteriors.plot_pair("mu", "sigma")


def test_posteriors_statistics(nll, model_params):
    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test statistical methods
    assert posteriors.median("mu").shape == ()
    assert posteriors.std("mu").shape == ()
    assert posteriors.mode("mu").shape == ()

    # Test covariance and correlation
    cov = posteriors.covariance()
    corr = posteriors.correlation()
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.abs(corr) <= 1.0)


def test_posteriors_predictive(nll, model_params, gauss):
    nwalkers = 5
    sampler = EmceeSampler(nwalkers=nwalkers)
    n_samples = 30
    posteriors = sampler.sample(loss=nll, n_samples=n_samples, n_warmup=10)

    # Test predictive distribution
    def predict_func():
        return gauss.sample(3)

    pred_samples = posteriors.predictive_distribution(predict_func)
    assert pred_samples.shape[0] == n_samples * nwalkers  # n_samples * nwalkers per sample


import numpy as np
import pytest
import tensorflow as tf
import zfit
from zfit._bayesian.results import Posterior

# Mock imports that might be used in tests
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for tests
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import corner

    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


@pytest.fixture
def sample_posterior():
    """Create a sample posterior for testing."""
    # Create parameters
    mu = zfit.Parameter("mean", 1.0)
    sigma = zfit.Parameter("sigma", 2.0)

    # Create sample MCMC data: 100 samples, 2 parameters
    samples = np.random.randn(100, 2)
    samples[:, 0] = samples[:, 0] + 1.0  # Center mu at 1.0
    samples[:, 1] = np.abs(samples[:, 1] * 0.5 + 2.0)  # Make sigma positive and center at 2.0

    # Create a simple loss function
    obs = zfit.Space("x", limits=(-10, 10))
    model = zfit.pdf.Gauss(mu, sigma, obs=obs)
    data = zfit.Data.from_numpy(obs=obs, array=np.random.randn(1000))
    loss = zfit.loss.UnbinnedNLL(model=model, data=data)

    # Create posterior
    posterior = Posterior(
        samples=samples, params=[mu, sigma], n_warmup=10, n_samples=100, loss=loss, sampler=None, raw_result=None
    )

    return posterior


class TestPosterior:
    """Tests for the Posterior class."""

    def test_initialization(self, sample_posterior):
        """Test that Posterior initializes correctly."""
        posterior = sample_posterior

        # Check basic attributes
        assert posterior.n_warmup == 10
        assert posterior.n_samples == 100
        assert len(posterior.samples) == 100
        assert posterior.samples.shape == (100, 2)
        assert posterior.param_names == ["mean", "sigma"]
        assert len(posterior.params) == 2
        assert posterior._param_index == {"mean": 0, "sigma": 1}

    def test_initialization_without_param_names(self):
        """Test initialization without explicitly providing param_names."""
        # Create parameters
        mu = zfit.Parameter("mean", 1.0)
        sigma = zfit.Parameter("sigma", 2.0)

        # Create sample data
        samples = np.random.randn(100, 2)

        # Create posterior without param_names
        posterior = Posterior(samples=samples, params=[mu, sigma], n_warmup=10, n_samples=100)

        # Check that param_names was derived correctly
        assert posterior.param_names == ["mean", "sigma"]

    def test_mean(self, sample_posterior):
        """Test mean calculation."""
        posterior = sample_posterior

        # Test mean for all parameters
        means = posterior.mean()
        assert isinstance(means, np.ndarray)
        assert len(means) == 2
        assert np.isclose(means[0], 1.0, atol=0.5)
        assert np.isclose(means[1], 2.0, atol=1.0)

        # Test mean for specific parameter by name
        mean_mu = posterior.mean("mean")
        assert np.isclose(mean_mu, means[0])

        # Test mean for specific parameter by object
        mean_sigma = posterior.mean(posterior.params[1])
        assert np.isclose(mean_sigma, means[1])

        # Test mean for specific parameter by index
        mean_mu_idx = posterior.mean(0)
        assert np.isclose(mean_mu_idx, means[0])

    def test_median(self, sample_posterior):
        """Test median calculation."""
        posterior = sample_posterior

        # Test median for all parameters
        medians = posterior.median()
        assert isinstance(medians, np.ndarray)
        assert len(medians) == 2

        # Test median for specific parameter
        median_mu = posterior.median("mean")
        assert np.isclose(median_mu, medians[0])

    def test_std(self, sample_posterior):
        """Test standard deviation calculation."""
        posterior = sample_posterior

        # Test std for all parameters
        stds = posterior.std()
        assert isinstance(stds, np.ndarray)
        assert len(stds) == 2
        assert np.all(stds > 0)  # Standard deviations should be positive

        # Test std for specific parameter
        std_sigma = posterior.std("sigma")
        assert np.isclose(std_sigma, stds[1])

    def test_mode(self, sample_posterior):
        """Test mode calculation."""
        posterior = sample_posterior

        # Test mode for all parameters
        modes = posterior.mode()
        assert isinstance(modes, np.ndarray)
        assert len(modes) == 2

        # Test mode for specific parameter with different bin settings
        mode_mu_50 = posterior.mode("mean", bins=50)
        mode_mu_200 = posterior.mode("mean", bins=200)
        # The two estimates might differ slightly but should be in the same ballpark
        assert np.isclose(mode_mu_50, mode_mu_200, atol=0.5)

    def test_credible_interval(self, sample_posterior):
        """Test credible interval calculation."""
        posterior = sample_posterior

        # Test credible interval for all parameters
        lower, upper = posterior.credible_interval(alpha=0.05)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(lower) == len(upper) == 2
        assert np.all(lower < upper)  # Lower bound should be less than upper bound

        # Test credible interval for specific parameter
        lower_mu, upper_mu = posterior.credible_interval("mean", alpha=0.1)
        # With alpha=0.1, we expect a 90% CI, which should be narrower than the 95% CI
        assert lower_mu > lower[0]
        assert upper_mu < upper[0]

        # Test invalid alpha
        with pytest.raises(ValueError):
            posterior.credible_interval(alpha=0)
        with pytest.raises(ValueError):
            posterior.credible_interval(alpha=1)

    def test_highest_density_interval(self, sample_posterior):
        """Test highest density interval calculation."""
        posterior = sample_posterior

        # Test HDI
        lower, upper = posterior.highest_density_interval("mean", alpha=0.05)
        assert lower < upper

        # Compare HDI with equal-tailed CI
        ci_lower, ci_upper = posterior.credible_interval("mean", alpha=0.05)
        hdi_width = upper - lower
        ci_width = ci_upper - ci_lower
        # HDI should be no wider than equal-tailed CI
        assert hdi_width <= ci_width * 1.05  # Allow 5% tolerance

    def test_sample(self, sample_posterior):
        """Test sample retrieval."""
        posterior = sample_posterior

        # Test getting all samples
        all_samples = posterior.sample()
        assert all_samples.shape == (100, 2)
        assert np.array_equal(all_samples, posterior.samples)

        # Test getting samples for specific parameter
        mu_samples = posterior.sample("mean")
        assert mu_samples.shape == (100,)
        assert np.array_equal(mu_samples, posterior.samples[:, 0])

    def test_covariance_correlation(self, sample_posterior):
        """Test covariance and correlation matrix calculation."""
        posterior = sample_posterior

        # Test covariance matrix
        cov = posterior.covariance()
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)  # Covariance matrix should be symmetric
        assert np.all(np.diag(cov) > 0)  # Diagonal elements (variances) should be positive

        # Test correlation matrix
        corr = posterior.correlation()
        assert corr.shape == (2, 2)
        assert np.allclose(corr, corr.T)  # Correlation matrix should be symmetric
        assert np.allclose(np.diag(corr), 1.0)  # Diagonal elements should be 1
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)  # Correlations between -1 and 1

    def test_summary(self, sample_posterior):
        """Test summary generation."""
        posterior = sample_posterior

        # Test summary dictionary
        summary = posterior.summary()
        assert set(summary.keys()) == set(posterior.param_names)

        # Check structure of summary for each parameter
        for param in posterior.param_names:
            param_summary = summary[param]
            expected_keys = {"mean", "median", "std", "ci_95_lower", "ci_95_upper"}
            assert set(param_summary.keys()) == expected_keys

    def test_print_summary(self, sample_posterior, capsys):
        """Test summary printing."""
        posterior = sample_posterior

        # Call print_summary and capture output
        posterior.print_summary()
        captured = capsys.readouterr()

        # Check that output contains key sections
        assert "Bayesian analysis summary" in captured.out
        assert "Parameter estimates" in captured.out
        assert "mean" in captured.out
        assert "sigma" in captured.out

    def test_param_index_lookup(self, sample_posterior):
        """Test parameter index lookup methods."""
        posterior = sample_posterior

        # Test lookup by string
        assert posterior._get_param_index("mean") == 0
        assert posterior._get_param_index("sigma") == 1

        # Test lookup by parameter object
        assert posterior._get_param_index(posterior.params[0]) == 0
        assert posterior._get_param_index(posterior.params[1]) == 1

        # Test lookup by index
        assert posterior._get_param_index(0) == 0
        assert posterior._get_param_index(1) == 1

        # Test invalid lookups
        with pytest.raises(ValueError):
            posterior._get_param_index("nonexistent")
        with pytest.raises(IndexError):
            posterior._get_param_index(2)  # Out of range
        with pytest.raises(TypeError):
            posterior._get_param_index(1.5)  # Wrong type

    def test_repr(self, sample_posterior):
        """Test string representation."""
        posterior = sample_posterior
        repr_str = repr(posterior)

        assert "Posterior" in repr_str
        assert "n_samples=100" in repr_str
        assert "mean" in repr_str
        assert "sigma" in repr_str

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not installed")
    def test_plot_trace(self, sample_posterior):
        """Test trace plotting."""
        posterior = sample_posterior
        ax = posterior.plot_trace("mean")
        assert ax is not None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not installed")
    def test_plot_posterior(self, sample_posterior):
        """Test posterior distribution plotting."""
        posterior = sample_posterior

        # Test with default settings
        ax = posterior.plot_posterior("mean")
        assert ax is not None

        # Test with custom settings
        ax = posterior.plot_posterior("sigma", hdi=False, show_point_estimates=False)
        assert ax is not None

    @pytest.mark.skipif(not HAS_CORNER, reason="Corner not installed")
    def test_plot_corner(self, sample_posterior):
        """Test corner plot."""
        posterior = sample_posterior
        fig = posterior.plot_corner()
        assert fig is not None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not installed")
    def test_plot_pair(self, sample_posterior):
        """Test pair plot."""
        posterior = sample_posterior

        # Test with default settings
        ax = posterior.plot_pair("mean", "sigma")
        assert ax is not None

        # Test with custom settings
        ax = posterior.plot_pair("mean", "sigma", contour=False, scatter=True)
        assert ax is not None

    def test_marginal_likelihood(self, sample_posterior):
        """Test marginal likelihood calculation."""
        posterior = sample_posterior

        # Skip the actual computation which is expensive and requires a proper model
        # Just test that the method exists and handles invalid method names
        with pytest.raises(ValueError):
            posterior.marginal_likelihood(method="invalid_method")

    def test_bayes_factor(self, sample_posterior):
        """Test Bayes factor calculation."""
        # Create two identical posteriors for testing
        posterior1 = sample_posterior
        posterior2 = sample_posterior

        # Mock the marginal_likelihood method to return predictable values
        original_ml_method = Posterior.marginal_likelihood
        try:
            Posterior.marginal_likelihood = lambda self, method="stepping", n_steps=20: -100.0

            # Test Bayes factor calculation
            bf = Posterior.bayes_factor(posterior1, posterior2)
            assert bf == 0.0  # log(1) = 0 since both have same evidence
        finally:
            # Restore original method
            Posterior.marginal_likelihood = original_ml_method


def test_posteriors_evidence(nll, model_params):
    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test evidence calculation methods
    evidence_stepping = posteriors.marginal_likelihood(method="stepping")
    evidence_harmonic = posteriors.marginal_likelihood(method="harmonic")

    assert np.isfinite(evidence_stepping)
    assert np.isfinite(evidence_harmonic)

    # Test Bayes factor
    posteriors2 = sampler.sample(loss=nll, n_samples=100, n_warmup=10)
    bf = posteriors.bayes_factor(posteriors, posteriors2)
    assert np.isfinite(bf)


def test_posteriors_parameter_access(nll, model_params):
    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test different ways to access parameters
    assert np.allclose(posteriors.mean("mu"), posteriors.mean(model_params["mu"]))
    assert np.allclose(posteriors.mean("mu"), posteriors.mean(0))

    # Test accessing all parameters at once
    means = posteriors.mean()
    assert means.shape == (2,)

    medians = posteriors.median()
    assert medians.shape == (2,)

    stds = posteriors.std()
    assert stds.shape == (2,)


def test_posteriors_invalid_inputs(nll, model_params):
    sampler = EmceeSampler()
    posteriors = sampler.sample(loss=nll, n_samples=100, n_warmup=10)

    # Test invalid inputs
    with pytest.raises(ValueError):
        posteriors.marginal_likelihood(method="invalid")

    with pytest.raises(ValueError):
        posteriors.credible_interval("mu", alpha=-0.1)

    with pytest.raises(ValueError):
        posteriors.credible_interval("mu", alpha=1.1)

    with pytest.raises(IndexError, match="Parameter index 123 out of range"):
        posteriors.mean(123)  # Invalid parameter index

#  Copyright (c) 2025 zfit

"""Performance tests for optimized accept_reject_sample function."""

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

import zfit
from zfit.core.sample import accept_reject_sample


@pytest.mark.parametrize("n_samples", [1, 1000, 10000])
def test_accept_reject_performance(n_samples):
    """Test performance and correctness of accept_reject_sample function."""
    # Create test PDF
    obs = zfit.Space("x", limits=(-5, 5))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    # Sample using optimized function
    samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n_samples, limits=obs, efficiency_estimation=0.5)

    # Basic correctness checks
    assert samples.shape == (n_samples, 1), f"Wrong shape: {samples.shape}"

    samples_np = samples.numpy().flatten()

    # Check samples are within limits
    assert np.all(samples_np >= -5), "Some samples below lower limit"
    assert np.all(samples_np <= 5), "Some samples above upper limit"

    # Statistical checks for larger samples
    if n_samples >= 1000:
        mean = np.mean(samples_np)
        std = np.std(samples_np)

        # Should be approximately normal(0, 1)
        assert abs(mean) < 0.2, f"Mean too far from 0: {mean}"
        assert 0.7 < std < 1.3, f"Std too far from 1: {std}"

        # K-S test for normality
        ks_stat, p_value = stats.kstest(samples_np, "norm")
        assert p_value > 0.01, f"K-S test failed: p={p_value}"


def test_small_sample_sizes():
    """Test with very small sample sizes."""
    obs = zfit.Space("x", limits=(-3, 3))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    for n in [1, 2, 5]:
        samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n, limits=obs, efficiency_estimation=0.5)

        assert samples.shape == (n, 1), f"Wrong shape for n={n}: {samples.shape}"

        # Check samples are within limits
        samples_np = samples.numpy()
        assert np.all(samples_np >= -3), f"Samples below limit for n={n}"
        assert np.all(samples_np <= 3), f"Samples above limit for n={n}"


@pytest.mark.parametrize("efficiency", [0.1, 0.5, 0.9])
def test_different_efficiency_estimations(efficiency):
    """Test with different efficiency estimations."""
    obs = zfit.Space("x", limits=(-5, 5))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    n = 1000
    samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n, limits=obs, efficiency_estimation=efficiency)

    assert samples.shape == (n, 1), f"Wrong shape for eff={efficiency}: {samples.shape}"

    # Basic statistical check
    samples_np = samples.numpy().flatten()
    mean = np.mean(samples_np)
    std = np.std(samples_np)

    # Should be roughly normal regardless of efficiency estimation
    assert abs(mean) < 0.3, f"Mean too far from 0 for eff={efficiency}: {mean}"
    assert 0.6 < std < 1.4, f"Std too far from 1 for eff={efficiency}: {std}"


def test_buffer_growth_scenario():
    """Test scenarios that require buffer growth."""
    # Create a PDF with low acceptance rate to force buffer growth
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 0.1)  # Very narrow Gaussian
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    # This should require buffer growth since initial 25% buffer won't be enough
    n = 5000
    samples = accept_reject_sample(
        prob=lambda x: pdf.pdf(x),
        n=n,
        limits=obs,
        efficiency_estimation=0.1,  # Low efficiency to force more draws
    )

    assert samples.shape == (n, 1), f"Wrong shape: {samples.shape}"

    # Check that samples are correctly distributed
    samples_np = samples.numpy().flatten()
    mean = np.mean(samples_np)
    std = np.std(samples_np)

    # Should be very narrow distribution
    assert abs(mean) < 0.05, f"Mean too far from 0: {mean}"
    assert 0.05 < std < 0.2, f"Std should be narrow: {std}"


def test_complex_pdf_mixture():
    """Test with a more complex PDF (mixture)."""
    obs = zfit.Space("x", limits=(-8, 8))

    # Create mixture of two Gaussians
    mu1 = zfit.Parameter("mu1", -2.0)
    sigma1 = zfit.Parameter("sigma1", 0.8)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)

    mu2 = zfit.Parameter("mu2", 3.0)
    sigma2 = zfit.Parameter("sigma2", 1.2)
    gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)

    frac = zfit.Parameter("frac", 0.4, 0, 1)
    pdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=frac)

    n = 5000
    samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n, limits=obs, efficiency_estimation=0.3)

    assert samples.shape == (n, 1), f"Wrong shape: {samples.shape}"

    samples_np = samples.numpy().flatten()

    # Basic sanity checks
    assert -8 <= samples_np.min(), "Samples below lower limit"
    assert samples_np.max() <= 8, "Samples above upper limit"

    # Should have some samples around both modes
    n_mode1 = np.sum((samples_np > -4) & (samples_np < 0))
    n_mode2 = np.sum((samples_np > 1) & (samples_np < 5))

    assert n_mode1 > n * 0.05, "Too few samples near first mode"
    assert n_mode2 > n * 0.05, "Too few samples near second mode"


def test_statistical_properties_large_sample():
    """Test statistical properties with larger samples."""
    obs = zfit.Space("x", limits=(-5, 5))
    mu = zfit.Parameter("mu", 1.5)
    sigma = zfit.Parameter("sigma", 0.7)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    n = 10000
    samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n, limits=obs, efficiency_estimation=0.5)

    samples_np = samples.numpy().flatten()

    # Statistical tests
    mean = np.mean(samples_np)
    std = np.std(samples_np)

    # Check mean and std are close to expected
    assert abs(mean - 1.5) < 0.1, f"Mean error too large: {abs(mean - 1.5)}"
    assert abs(std - 0.7) < 0.1, f"Std error too large: {abs(std - 0.7)}"

    # K-S test against the expected normal distribution
    ks_stat, p_value = stats.kstest(samples_np, lambda x: stats.norm.cdf(x, loc=1.5, scale=0.7))
    assert p_value > 0.01, f"K-S test failed: p={p_value}"


@pytest.mark.benchmark
def test_performance_scaling():
    """Test that performance scales well with sample size."""
    import time

    obs = zfit.Space("x", limits=(-5, 5))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    sizes = [1000, 5000, 10000]
    times = []

    for n in sizes:
        # Warm up
        _ = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=100, limits=obs)

        # Time the sampling
        start_time = time.time()
        samples = accept_reject_sample(prob=lambda x: pdf.pdf(x), n=n, limits=obs, efficiency_estimation=0.5)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        # Verify correctness
        assert samples.shape == (n, 1)

    # Check that scaling is sub-linear (should be near-constant with optimization)
    # Calculate time complexity exponent
    log_n = np.log(sizes)
    log_t = np.log(times)
    slope, _ = np.polyfit(log_n, log_t, 1)

    # With optimization, should have excellent scaling (< 0.5 is very good)
    assert slope < 1.0, f"Scaling too poor: O(n^{slope:.3f}), times: {times}"

    # Calculate throughput
    throughputs = [n / t for n, t in zip(sizes, times)]
    min_throughput = min(throughputs)

    # Should achieve reasonable throughput
    assert min_throughput > 1000, f"Throughput too low: {min_throughput} samples/sec"

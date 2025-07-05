#!/usr/bin/env python3
"""Test KDE performance between manual and TFP implementations."""

import numpy as np
import pytest
import tensorflow as tf
import zfit
import time


class KDE1DimExactTFP(zfit.pdf.KDE1DimExact):
    """KDE with TFP implementation for comparison."""

    def _unnormalized_pdf(self, x):
        """Use the standard TFP implementation."""
        return super(zfit.pdf.KDE1DimExact, self)._unnormalized_pdf(x)


def test_kde_manual_vs_tfp_correctness():
    """Test that manual and TFP implementations give the same results."""
    # Setup test data
    np.random.seed(42)
    n_data = 100
    n_test = 1000

    limits = (-5, 5)
    obs = zfit.Space("obs1", limits=limits)

    # Generate random data
    data_vals = np.random.normal(0, 1, (n_data, 1))
    weights = np.random.uniform(0.5, 1.5, n_data)

    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights)

    # Create both KDE versions
    kde_manual = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
    kde_tfp = KDE1DimExactTFP(data, bandwidth='silverman')

    # Test points
    test_x = np.linspace(-4, 4, n_test).reshape(-1, 1)

    # Get PDF values
    pdf_manual = kde_manual.pdf(test_x, norm=False).numpy()
    pdf_tfp = kde_tfp.pdf(test_x, norm=False).numpy()

    # They should be very close
    np.testing.assert_allclose(pdf_manual, pdf_tfp, rtol=1e-10, atol=1e-12)

    # Test with no weights
    data_no_weights = zfit.data.Data.from_numpy(obs=obs, array=data_vals)

    kde_manual_nw = zfit.pdf.KDE1DimExact(data_no_weights, bandwidth='silverman')
    kde_tfp_nw = KDE1DimExactTFP(data_no_weights, bandwidth='silverman')

    pdf_manual_nw = kde_manual_nw.pdf(test_x, norm=False).numpy()
    pdf_tfp_nw = kde_tfp_nw.pdf(test_x, norm=False).numpy()

    np.testing.assert_allclose(pdf_manual_nw, pdf_tfp_nw, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n_data,n_eval", [
    (100, 1000),
    (1000, 100),
    (1000, 100000),
])
def test_kde_manual_vs_tfp_performance(n_data, n_eval):
    """Compare performance of manual vs TFP implementations."""
    # Setup
    np.random.seed(42)
    limits = (-5, 5)
    obs = zfit.Space("obs1", limits=limits)

    # Generate data
    data_vals = np.random.normal(0, 1, (n_data, 1))
    weights = np.random.uniform(0.5, 1.5, n_data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights)

    # Create KDEs
    kde_manual = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
    kde_tfp = KDE1DimExactTFP(data, bandwidth='silverman')

    # Test points
    test_x = np.linspace(-4, 4, n_eval).reshape(-1, 1)

    # Warm up
    _ = kde_manual.pdf(test_x[:10], norm=False).numpy()
    _ = kde_tfp.pdf(test_x[:10], norm=False).numpy()

    # Time manual implementation
    n_runs = 10
    manual_times = []
    for _ in range(n_runs):
        start = time.time()
        _ = kde_manual.pdf(test_x, norm=False).numpy()
        manual_times.append(time.time() - start)

    # Time TFP implementation
    tfp_times = []
    for _ in range(n_runs):
        start = time.time()
        _ = kde_tfp.pdf(test_x, norm=False).numpy()
        tfp_times.append(time.time() - start)

    manual_mean = np.mean(manual_times)
    tfp_mean = np.mean(tfp_times)


    # Manual should be within reasonable performance of TFP (not more than 3x slower)
    assert manual_mean < tfp_mean * 3.0, f"Manual implementation is too slow: {manual_mean/tfp_mean:.2f}x slower"


def test_kde_manual_with_array_bandwidth():
    """Test manual implementation with array bandwidth."""
    # Setup
    np.random.seed(42)
    n_data = 50

    limits = (-5, 5)
    obs = zfit.Space("obs1", limits=limits)

    # Generate data
    data_vals = np.random.normal(0, 1, (n_data, 1))
    weights = np.random.uniform(0.5, 1.5, n_data)

    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights)

    # Create array bandwidth
    bandwidth_array = np.random.uniform(0.1, 0.5, n_data)

    # Create KDEs
    kde_manual = zfit.pdf.KDE1DimExact(data, bandwidth=bandwidth_array)
    kde_tfp = KDE1DimExactTFP(data, bandwidth=bandwidth_array)

    # Test points
    test_x = np.linspace(-4, 4, 100).reshape(-1, 1)

    # Get PDF values
    pdf_manual = kde_manual.pdf(test_x, norm=False).numpy()
    pdf_tfp = kde_tfp.pdf(test_x, norm=False).numpy()

    # They should be very close
    np.testing.assert_allclose(pdf_manual, pdf_tfp, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    test_kde_manual_vs_tfp_correctness()
    test_kde_manual_with_array_bandwidth()
    for n_data, n_eval in [(100, 1000), (1000, 100), (1000, 1000)]:
        test_kde_manual_vs_tfp_performance(n_data, n_eval)

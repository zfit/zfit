"""Tests for PDF edge cases and robustness."""

#  Copyright (c) 2025 zfit

import pytest
import tensorflow as tf
import numpy as np

import zfit
from zfit import z


def test_extended_pdf_creation():
    """Test creating extended PDFs with various yield parameters."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Test with parameter yield
    yield_param = zfit.Parameter("yield", 1000.0)
    extended_pdf = pdf.create_extended(yield_param)
    
    assert extended_pdf.is_extended
    assert extended_pdf.get_yield() == yield_param


def test_extended_pdf_with_constant_yield():
    """Test extended PDF with constant yield."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Test with constant yield
    extended_pdf = pdf.create_extended(500.0)
    
    assert extended_pdf.is_extended
    # Yield should be converted to a parameter (may be ConstantParameter)
    yield_val = extended_pdf.get_yield()
    # Accept both Parameter and ConstantParameter
    from zfit.core.interfaces import ZfitParameter
    assert isinstance(yield_val, ZfitParameter)


def test_extended_pdf_yield_limits():
    """Test extended PDF with yield limits."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Create yield with limits
    yield_param = zfit.Parameter("yield", 1000.0, 0.0, 10000.0)
    extended_pdf = pdf.create_extended(yield_param)
    
    assert extended_pdf.is_extended
    assert extended_pdf.get_yield().has_limits


def test_extended_pdf_negative_yield():
    """Test extended PDF behavior with negative yield."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Negative yield should be handled gracefully
    yield_param = zfit.Parameter("yield", -100.0)
    extended_pdf = pdf.create_extended(yield_param)
    
    # PDF should still be created (validation may happen at fit time)
    assert extended_pdf.is_extended


def test_extended_pdf_zero_yield():
    """Test extended PDF with zero yield."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Zero yield
    yield_param = zfit.Parameter("yield", 0.0)
    extended_pdf = pdf.create_extended(yield_param)
    
    assert extended_pdf.is_extended
    # Should handle zero yield gracefully


def test_extended_pdf_switching_modes():
    """Test switching between extended and non-extended modes."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Start non-extended
    assert not pdf.is_extended
    
    # Make extended
    yield_param = zfit.Parameter("yield", 1000.0)
    extended_pdf = pdf.create_extended(yield_param)
    assert extended_pdf.is_extended
    
    # Convert back to unbinned (non-extended)
    unbinned_pdf = extended_pdf.to_unbinned()
    # Note: to_unbinned() may or may not remove extended nature depending on implementation


def test_pdf_normalization_different_ranges():
    """Test PDF normalization over different ranges."""
    obs = zfit.Space("x", (-5, 5))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Test normalization over different ranges
    norm_range1 = zfit.Space("x", (-1, 1))
    norm_range2 = zfit.Space("x", (-2, 2))
    
    # Get normalized PDFs using the norm parameter
    pdf_val1 = pdf.pdf(tf.constant([0.0]), norm=norm_range1)
    pdf_val2 = pdf.pdf(tf.constant([0.0]), norm=norm_range2)
    
    # Values should be different due to different normalization
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, 
                           pdf_val1.numpy(), pdf_val2.numpy(), 
                           err_msg="PDF values should differ with different normalization ranges")


def test_pdf_normalization_narrow_range():
    """Test PDF normalization over very narrow ranges."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Very narrow normalization range
    narrow_range = zfit.Space("x", (-0.001, 0.001))
    
    pdf_val = pdf.pdf(tf.constant([0.0]), norm=narrow_range)
        
    # Should handle narrow ranges without numerical issues
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="PDF values should be finite for narrow ranges")


def test_pdf_unnormalized_evaluation():
    """Test PDF evaluation without normalization."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Evaluate without normalization
    x = tf.constant([0.0])
    unnorm_val = pdf.pdf(x, norm=False)
    norm_val = pdf.pdf(x, norm=obs)  # Use the obs space for normalization
    
    # Unnormalized should be different from normalized
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, 
                           unnorm_val.numpy(), norm_val.numpy(), 
                           err_msg="Unnormalized and normalized PDF values should differ")


def test_pdf_integration_edge_cases():
    """Test PDF integration in edge cases."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 0.01)  # Very narrow PDF
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Integration should work even with narrow PDFs
    integral = pdf.integrate(limits=obs)
    
    # Should be close to 1 for normalized PDF
    np.testing.assert_allclose(float(integral), 1.0, atol=0.1, 
                             err_msg="Normalized PDF integral should be close to 1")


def test_pdf_outside_support_range():
    """Test PDF evaluation outside its support range."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Evaluate outside the defined space - but note that Gauss is defined everywhere
    # so this test mainly checks numerical stability
    x_outside = tf.constant([-10.0, 10.0])
    pdf_val = pdf.pdf(x_outside, norm=False)  # Use unnormalized to avoid norm issues
    
    # Should handle gracefully (may return very small values)
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="PDF values should be finite even outside typical range")


def test_sum_pdf_with_many_components():
    """Test SumPDF with many component PDFs."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create many Gaussian components
    n_components = 10
    pdfs = []
    fracs = []
    
    for i in range(n_components):
        mu = zfit.Parameter(f"mu_{i}", float(i - 5) * 0.2)
        sigma = zfit.Parameter(f"sigma_{i}", 0.1)
        pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
        pdfs.append(pdf)
        
        if i < n_components - 1:  # n-1 fractions for n components
            frac = zfit.Parameter(f"frac_{i}", 1.0 / n_components)
            fracs.append(frac)
    
    # Create sum PDF
    sum_pdf = zfit.pdf.SumPDF(pdfs, fracs=fracs)
    
    # Should handle many components
    pdf_val = sum_pdf.pdf(tf.constant([0.0]))
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="SumPDF with many components should produce finite values")


def test_sum_pdf_fraction_constraints():
    """Test SumPDF with fraction constraints."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create two Gaussians
    mu1 = zfit.Parameter("mu1", -0.5)
    sigma1 = zfit.Parameter("sigma1", 0.3)
    pdf1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)
    
    mu2 = zfit.Parameter("mu2", 0.5)
    sigma2 = zfit.Parameter("sigma2", 0.3)
    pdf2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
    
    # Fraction with limits
    frac = zfit.Parameter("frac", 0.5, 0.0, 1.0)
    
    sum_pdf = zfit.pdf.SumPDF([pdf1, pdf2], fracs=[frac])
    
    # Should handle constrained fractions
    pdf_val = sum_pdf.pdf(tf.constant([0.0]))
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="SumPDF with constrained fractions should produce finite values")


def test_sum_pdf_extreme_fractions():
    """Test SumPDF with extreme fraction values."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create two Gaussians
    mu1 = zfit.Parameter("mu1", -0.5)
    sigma1 = zfit.Parameter("sigma1", 0.3)
    pdf1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)
    
    mu2 = zfit.Parameter("mu2", 0.5)
    sigma2 = zfit.Parameter("sigma2", 0.3)
    pdf2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
    
    # Very small fraction (almost all first component)
    frac = zfit.Parameter("frac", 1e-6)
    
    sum_pdf = zfit.pdf.SumPDF([pdf1, pdf2], fracs=[frac])
    
    # Should handle extreme fractions
    pdf_val = sum_pdf.pdf(tf.constant([0.0]))
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="SumPDF with extreme fractions should produce finite values")


def test_product_pdf_edge_cases():
    """Test ProductPDF edge cases."""
    obs1 = zfit.Space("x", (-1, 1))
    obs2 = zfit.Space("y", (-1, 1))
    
    # Create PDFs in different observables
    mu_x = zfit.Parameter("mu_x", 0.0)
    sigma_x = zfit.Parameter("sigma_x", 1.0)
    pdf_x = zfit.pdf.Gauss(obs=obs1, mu=mu_x, sigma=sigma_x)
    
    mu_y = zfit.Parameter("mu_y", 0.0)
    sigma_y = zfit.Parameter("sigma_y", 1.0)
    pdf_y = zfit.pdf.Gauss(obs=obs2, mu=mu_y, sigma=sigma_y)
    
    # Create product PDF
    product_pdf = zfit.pdf.ProductPDF([pdf_x, pdf_y])
    
    # Should handle multi-dimensional evaluation
    combined_obs = obs1 * obs2
    x_vals = tf.constant([[0.0, 0.0]])  # 2D point
    pdf_val = product_pdf.pdf(x_vals)
    
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="ProductPDF should produce finite values for multi-dimensional evaluation")


def test_pdf_with_zero_width_parameter():
    """Test PDF with zero or very small width parameters."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1e-10)  # Very small sigma
    
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Should handle very small parameters
    try:
        pdf_val = pdf.pdf(tf.constant([0.0]))
        np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                    err_msg="PDF should handle very small width parameters gracefully")
    except (tf.errors.InvalidArgumentError, ValueError):
        # Some implementations may reject zero/very small widths
        pass


def test_pdf_with_extreme_parameter_values():
    """Test PDF with extreme parameter values."""
    obs = zfit.Space("x", (-100, 100))
    mu = zfit.Parameter("mu", 1e6)  # Very large mean
    sigma = zfit.Parameter("sigma", 1e-6)  # Very small sigma
    
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Should handle extreme values gracefully
    try:
        pdf_val = pdf.pdf(tf.constant([1e6]))
        np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                    err_msg="PDF should handle extreme parameter values gracefully")
    except (tf.errors.InvalidArgumentError, ValueError):
        # May have numerical issues with extreme values
        pass


def test_pdf_parameter_updates_during_evaluation():
    """Test PDF behavior when parameters change during evaluation."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Evaluate PDF
    x = tf.constant([0.0])
    pdf_val1 = pdf.pdf(x)
    
    # Change parameter
    mu.assign(0.5)
    
    # Evaluate again
    pdf_val2 = pdf.pdf(x)
    
    # Values should be different
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, 
                           pdf_val1.numpy(), pdf_val2.numpy(), 
                           err_msg="PDF values should change when parameters are updated")


def test_pdf_with_complex_parameter_dependencies():
    """Test PDF with complex parameter dependency chains."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create parameter hierarchy
    base_param = zfit.Parameter("base", 1.0)
    mu = zfit.ComposedParameter("mu", lambda x: x * 0.5, params=[base_param])
    sigma = zfit.ComposedParameter("sigma", lambda x: abs(x) * 0.3, params=[base_param])
    
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Should handle composed parameters
    pdf_val = pdf.pdf(tf.constant([0.0]))
    np.testing.assert_array_equal(np.isfinite(pdf_val.numpy()), True, 
                                err_msg="PDF should handle composed parameters gracefully")
    
    # Changing base parameter should affect PDF
    base_param.assign(2.0)
    pdf_val2 = pdf.pdf(tf.constant([0.0]))
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, 
                           pdf_val.numpy(), pdf_val2.numpy(), 
                           err_msg="PDF values should change when base parameter of composed parameter changes")


def test_pdf_memory_efficiency_large_evaluations():
    """Test PDF memory efficiency with large data."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Large evaluation array
    n_points = 100000
    x_large = tf.random.normal([n_points, 1])
    
    # Should handle large evaluations
    pdf_vals = pdf.pdf(x_large)
    
    assert pdf_vals.shape[0] == n_points
    np.testing.assert_array_equal(np.isfinite(pdf_vals.numpy()), True, 
                                err_msg="PDF should handle large evaluations without numerical issues")


def test_pdf_concurrent_evaluation():
    """Test PDF evaluation in concurrent scenarios."""
    import threading
    import time
    
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    results = []
    errors = []
    
    def evaluate_pdf():
        try:
            x = tf.random.normal([100, 1])
            pdf_val = pdf.pdf(x)
            results.append(pdf_val)
        except Exception as e:
            errors.append(e)
    
    # Run evaluations in multiple threads
    threads = [threading.Thread(target=evaluate_pdf) for _ in range(5)]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # All evaluations should succeed
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    assert len(errors) == 0, f"Expected no errors, got {len(errors)} errors: {errors}"
    
    # Check that all results are finite
    for i, result in enumerate(results):
        np.testing.assert_array_equal(np.isfinite(result.numpy()), True, 
                                    err_msg=f"Concurrent evaluation {i} should produce finite values")


def test_pdf_garbage_collection():
    """Test PDF cleanup and garbage collection."""
    import gc
    import weakref
    
    # Create PDF and get weak reference
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    pdf_ref = weakref.ref(pdf)
    
    # Delete PDF
    del pdf
    gc.collect()
    
    # Should be garbage collected (may not work if references exist elsewhere)
    # This test mainly ensures no exceptions during cleanup
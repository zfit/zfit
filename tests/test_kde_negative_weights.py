#!/usr/bin/env python3
"""
Test negative weights functionality for KDE PDFs.

This test ensures that KDE can handle negative weights without producing NaN values.
The test covers the issue described in GitHub issue #620.
"""

import numpy as np
# import pytest  # Commented out for direct execution
import zfit


class TestNegativeWeightsKDE:
    """Test class for KDE with negative weights."""

    def test_kde_negative_weights_basic(self):
        """Test basic negative weights functionality for KDE1DimExact."""
        obs = zfit.Space('x', limits=(-4, +4))
        
        # Create data with mixed positive and negative weights
        data_vals = np.array([0.0, 1.0, 2.0])
        weights = np.array([1.0, 1.0, -0.1])  # One negative weight
        
        data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
        
        # Test PDF evaluation
        test_x = np.array([0.0, 1.0, 2.0])
        pdf_vals = pdf.pdf(test_x).numpy()
        
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(pdf_vals))
        assert not np.any(np.isinf(pdf_vals))
        assert np.all(pdf_vals >= 0)  # PDFs should be non-negative

    def test_kde_negative_weights_original_reproducer(self):
        """Test the original reproducer from GitHub issue #620."""
        obs = zfit.Space('x', limits=(-4, +4))
        
        # Recreate the original issue scenario
        nentries = 500
        arr_val = np.random.normal(loc=0, scale=1.0, size=nentries)
        arr_wgt = np.random.normal(loc=1, scale=0.1, size=nentries)
        
        # Add negative weight entry
        arr_val = np.concatenate([arr_val, [0.0]])
        arr_wgt = np.concatenate([arr_wgt, [-0.01]])
        
        data = zfit.data.Data.from_numpy(obs=obs, array=arr_val.reshape(-1, 1), weights=arr_wgt)
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='isj')
        
        # Test PDF evaluation
        test_x = np.linspace(-4, +4, 20)
        pdf_vals = pdf.pdf(test_x).numpy()
        
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(pdf_vals))
        assert not np.any(np.isinf(pdf_vals))
        assert np.all(pdf_vals >= 0)

    # @pytest.mark.parametrize("kde_class,kwargs", [
    #     (zfit.pdf.KDE1DimExact, {'bandwidth': 'silverman'}),
    #     (zfit.pdf.KDE1DimGrid, {'bandwidth': 'silverman'}),
    #     (zfit.pdf.KDE1DimFFT, {'bandwidth': 'silverman'}),
    #     (zfit.pdf.KDE1DimISJ, {}),
    # ])
    def test_all_kde_variants_negative_weights(self, kde_class, kwargs):
        """Test all KDE variants with negative weights."""
        obs = zfit.Space('x', limits=(-4, +4))
        
        # Create data with some negative weights
        np.random.seed(42)  # For reproducible results
        data_vals = np.random.normal(0, 1, 50)
        weights = np.random.normal(1, 0.1, 50)
        
        # Add some negative weights
        weights[45:] = -0.01  # Last 5 weights are negative
        
        data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
        kde = kde_class(data=data, **kwargs)
        
        # Test PDF evaluation
        test_x = np.linspace(-2, 2, 10)
        pdf_vals = kde.pdf(test_x).numpy()
        
        # Should not contain NaN or Inf  
        assert not np.any(np.isnan(pdf_vals))
        assert not np.any(np.isinf(pdf_vals))
        assert np.all(pdf_vals >= 0)
        
        # Test integration (should be close to 1)
        integral = kde.integrate(limits=obs).numpy()
        assert abs(float(integral) - 1.0) < 0.01
        
        # Test sampling
        sample = kde.sample(5).numpy()
        assert sample.shape == (5, 1)

    def test_kde_all_negative_weights(self):
        """Test KDE when all weights are negative."""
        obs = zfit.Space('x', limits=(-4, +4))
        
        # All negative weights
        data_vals = np.array([0.0, 1.0, 2.0])
        weights = np.array([-1.0, -0.5, -0.3])
        
        data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
        
        # Test PDF evaluation
        test_x = np.array([0.0, 1.0, 2.0])
        pdf_vals = pdf.pdf(test_x).numpy()
        
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(pdf_vals))
        assert not np.any(np.isinf(pdf_vals))
        assert np.all(pdf_vals >= 0)

    def test_kde_zero_sum_weights(self):
        """Test KDE when weights sum to zero."""
        obs = zfit.Space('x', limits=(-4, +4))
        
        # Weights that sum to zero
        data_vals = np.array([0.0, 1.0, 2.0])
        weights = np.array([1.0, 1.0, -2.0])
        
        data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
        
        # Test PDF evaluation
        test_x = np.array([0.0, 1.0, 2.0])
        pdf_vals = pdf.pdf(test_x).numpy()
        
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(pdf_vals))
        assert not np.any(np.isinf(pdf_vals))
        assert np.all(pdf_vals >= 0)


if __name__ == "__main__":
    # Run tests directly if executed as script
    test_instance = TestNegativeWeightsKDE()
    
    print("Running KDE negative weights tests...")
    
    try:
        test_instance.test_kde_negative_weights_basic()
        print("✓ Basic negative weights test passed")
    except Exception as e:
        print(f"✗ Basic negative weights test failed: {e}")
    
    try:
        test_instance.test_kde_negative_weights_original_reproducer()
        print("✓ Original reproducer test passed")
    except Exception as e:
        print(f"✗ Original reproducer test failed: {e}")
    
    try:
        test_instance.test_kde_all_negative_weights()
        print("✓ All negative weights test passed")
    except Exception as e:
        print(f"✗ All negative weights test failed: {e}")
        
    try:
        test_instance.test_kde_zero_sum_weights()
        print("✓ Zero sum weights test passed")
    except Exception as e:
        print(f"✗ Zero sum weights test failed: {e}")
    
    # Test all variants
    variants = [
        (zfit.pdf.KDE1DimExact, {'bandwidth': 'silverman'}),
        (zfit.pdf.KDE1DimGrid, {'bandwidth': 'silverman'}),
        (zfit.pdf.KDE1DimFFT, {'bandwidth': 'silverman'}),
        (zfit.pdf.KDE1DimISJ, {}),
    ]
    
    for kde_class, kwargs in variants:
        try:
            test_instance.test_all_kde_variants_negative_weights(kde_class, kwargs)
            print(f"✓ {kde_class.__name__} negative weights test passed")
        except Exception as e:
            print(f"✗ {kde_class.__name__} negative weights test failed: {e}")
    
    print("\nAll tests completed!")
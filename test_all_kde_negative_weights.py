#!/usr/bin/env python3
"""Test negative weights with all KDE variants."""

import zfit
import numpy as np
import pandas as pd

def test_kde_variant(kde_class, kde_name, **kwargs):
    """Test a specific KDE variant with negative weights."""
    print(f"\n=== Testing {kde_name} ===")
    
    try:
        # Create test data with negative weights
        obs = zfit.Space('x', limits=(-4, +4))
        
        # Mix of positive and negative weights
        np.random.seed(42)  # For reproducible results
        data_vals = np.random.normal(0, 1, 100) 
        weights = np.random.normal(1, 0.1, 100)
        
        # Add some negative weights
        weights[90:95] = -0.01  # 5 negative weights
        
        print(f"Number of negative weights: {sum(weights < 0)}")
        print(f"Sum of all weights: {np.sum(weights):.4f}")
        
        data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
        
        # Create KDE
        kde = kde_class(data=data, **kwargs)
        
        # Test PDF evaluation
        test_x = np.linspace(-2, 2, 10)
        pdf_vals = kde.pdf(test_x).numpy()
        
        # Check results
        has_nan = np.any(np.isnan(pdf_vals))
        has_inf = np.any(np.isinf(pdf_vals))
        has_negative = np.any(pdf_vals < 0)
        min_val = np.min(pdf_vals)
        max_val = np.max(pdf_vals)
        
        print(f"✓ PDF evaluation successful")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  Has negative: {has_negative}")
        print(f"  Min value: {min_val:.2e}")
        print(f"  Max value: {max_val:.2e}")
        
        # Test integration
        integral = kde.integrate(limits=obs).numpy()
        print(f"  Integral: {float(integral):.4f}")
        
        # Test sampling
        sample = kde.sample(5).numpy()
        print(f"  Sample shape: {sample.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all KDE variants with negative weights."""
    print("Testing KDE variants with negative weights")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    # Test KDE1DimExact
    total_count += 1
    if test_kde_variant(zfit.pdf.KDE1DimExact, "KDE1DimExact", bandwidth='silverman'):
        success_count += 1
    
    # Test KDE1DimGrid  
    total_count += 1
    if test_kde_variant(zfit.pdf.KDE1DimGrid, "KDE1DimGrid", bandwidth='silverman'):
        success_count += 1
    
    # Test KDE1DimFFT
    total_count += 1
    if test_kde_variant(zfit.pdf.KDE1DimFFT, "KDE1DimFFT", bandwidth='silverman'):
        success_count += 1
    
    # Test KDE1DimISJ
    total_count += 1  
    if test_kde_variant(zfit.pdf.KDE1DimISJ, "KDE1DimISJ"):
        success_count += 1
    
    # Test GaussianKDE1DimV1 (requires obs parameter)
    total_count += 1
    obs_for_gauss = zfit.Space('x', limits=(-4, +4))
    if test_kde_variant(zfit.pdf.GaussianKDE1DimV1, "GaussianKDE1DimV1", obs=obs_for_gauss, bandwidth='silverman'):
        success_count += 1
    
    print(f"\n" + "=" * 50)
    print(f"Summary: {success_count}/{total_count} KDE variants passed")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == '__main__':
    main()
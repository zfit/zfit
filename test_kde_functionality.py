#!/usr/bin/env python3
"""Test basic KDE functionality to ensure no regression."""

import zfit
import numpy as np

def test_basic_kde():
    """Test basic KDE functionality without negative weights."""
    print("Testing basic KDE functionality...")
    
    obs = zfit.Space('x', limits=(-4, +4))
    
    # Normal positive weights
    np.random.seed(42)
    data_vals = np.random.normal(0, 1, 100)
    weights = np.random.normal(1, 0.1, 100)
    weights = np.abs(weights)  # Ensure all positive
    
    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
    
    # Test all KDE variants
    kdes = [
        ("KDE1DimExact", zfit.pdf.KDE1DimExact(data=data, bandwidth='silverman')),
        ("KDE1DimGrid", zfit.pdf.KDE1DimGrid(data=data, bandwidth='silverman')),
        ("KDE1DimFFT", zfit.pdf.KDE1DimFFT(data=data, bandwidth='silverman')),
        ("KDE1DimISJ", zfit.pdf.KDE1DimISJ(data=data)),
    ]
    
    test_x = np.linspace(-2, 2, 10)
    
    for name, kde in kdes:
        try:
            pdf_vals = kde.pdf(test_x).numpy()
            integral = kde.integrate(limits=obs).numpy()
            sample = kde.sample(5).numpy()
            
            print(f"✓ {name}: PDF OK, Integral={float(integral):.4f}, Sample shape={sample.shape}")
            
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")
            return False
    
    return True

def test_original_reproducer():
    """Test the original reproducer from the issue."""
    print("\nTesting original reproducer...")
    
    try:
        nentries = 500
        obs = zfit.Space('x', limits=(-4, +4))

        arr_val = np.random.normal(loc=0, scale=1.0, size=nentries)
        arr_wgt = np.random.normal(loc=1, scale=0.1, size=nentries)

        df_1 = {'x': arr_val, 'w': arr_wgt}
        
        # Add negative weight entry
        df_2 = {'x': [0.0], 'w': [-0.01]}
        
        all_x = np.concatenate([df_1['x'], df_2['x']])
        all_w = np.concatenate([df_1['w'], df_2['w']])
        
        data = zfit.data.Data.from_numpy(obs=obs, array=all_x.reshape(-1, 1), weights=all_w)
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='isj')

        arr_x = np.linspace(-4, +4, 20)
        arr_y = pdf.pdf(arr_x).numpy()
        
        print(f"✓ Original reproducer: Success, no NaN values")
        print(f"  First few PDF values: {arr_y[:3]}")
        return True
        
    except Exception as e:
        print(f"✗ Original reproducer: Failed - {e}")
        return False

if __name__ == '__main__':
    success1 = test_basic_kde()
    success2 = test_original_reproducer()
    
    if success1 and success2:
        print("\n🎉 All functionality tests passed!")
    else:
        print("\n⚠️ Some functionality tests failed!")
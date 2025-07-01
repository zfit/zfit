#!/usr/bin/env python3
"""Test to understand where NaN occurs in the KDE calculation."""

import zfit
import numpy as np
import tensorflow as tf

def test_kde_step_by_step():
    print("=== Step-by-step KDE debugging ===")
    
    # Simple test case
    obs = zfit.Space('x', limits=(-4, +4))
    data_vals = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, -0.1])
    
    print(f"Data values: {data_vals}")
    print(f"Weights: {weights}")
    
    # Create data
    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
    
    # Create KDE
    pdf = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
    
    # Test the unnormalized PDF directly
    test_x = np.array([0.0, 1.0, 2.0])
    print(f"Test x values: {test_x}")
    
    # Call the unnormalized PDF directly
    try:
        unnorm_pdf = pdf._unnormalized_pdf(test_x)
        print(f"Unnormalized PDF values: {unnorm_pdf}")
        print(f"Min unnormalized PDF value: {tf.reduce_min(unnorm_pdf)}")
        print(f"Has NaN in unnormalized PDF: {tf.reduce_any(tf.math.is_nan(unnorm_pdf))}")
    except Exception as e:
        print(f"Error in unnormalized PDF: {e}")
        import traceback
        traceback.print_exc()
    
    # Try the full PDF
    try:
        full_pdf = pdf.pdf(test_x)
        print(f"Full PDF values: {full_pdf}")
    except Exception as e:
        print(f"Error in full PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_kde_step_by_step()
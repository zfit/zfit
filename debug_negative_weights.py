#!/usr/bin/env python3
"""Debug script to understand the negative weights issue with KDE."""

import zfit
import numpy as np
import pandas as pd
import tensorflow as tf

# Set up debugging
import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '0'

def test_simple_negative_weights():
    print("=== Testing Simple Negative Weights ===")
    # Create simple test data
    obs = zfit.Space('x', limits=(-4, +4))
    
    # Mix of positive and negative weights
    data_vals = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, -0.1])  # One negative weight
    
    print(f"Data values: {data_vals}")
    print(f"Weights: {weights}")
    print(f"Sum of weights: {np.sum(weights)}")
    
    # Create data
    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals.reshape(-1, 1), weights=weights)
    
    try:
        # Create KDE
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
        
        # Test evaluation
        test_x = np.array([0.0, 1.0, 2.0])
        print(f"Test x values: {test_x}")
        
        # Try to evaluate PDF
        pdf_vals = pdf.pdf(test_x)
        print(f"PDF values: {pdf_vals.numpy()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_original_reproducer():
    print("\n=== Testing Original Reproducer ===")
    try:
        nentries = 500
        obs = zfit.Space('x', limits=(-4, +4))

        # Positive weights data
        arr_val_1 = np.random.normal(loc=0, scale=1.0, size=nentries)
        arr_wgt_1 = np.random.normal(loc=1, scale=0.1, size=nentries)
        
        # Negative weights data (smaller)
        arr_val_2 = np.random.normal(loc=0, scale=1.0, size=1)
        arr_wgt_2 = np.random.normal(loc=-0.01, scale=0.001, size=1)

        df_1 = pd.DataFrame({'x': arr_val_1, 'w': arr_wgt_1})
        df_2 = pd.DataFrame({'x': arr_val_2, 'w': arr_wgt_2})

        df = pd.concat([df_1, df_2])
        print(f"Total entries: {len(df)}")
        print(f"Negative weight entries: {len(df[df.w < 0])}")
        print(f"Sum of all weights: {df.w.sum()}")

        data = zfit.data.Data.from_pandas(df=df, obs=obs, weights='w')
        pdf = zfit.pdf.KDE1DimExact(data, bandwidth='isj')

        arr_x = np.linspace(-4, +4, 20)
        arr_y = pdf.pdf(arr_x).numpy()
        print(f"PDF evaluation successful: {not np.any(np.isnan(arr_y))}")
        print(f"First few PDF values: {arr_y[:5]}")
        
    except Exception as e:
        print(f"Error in original reproducer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_simple_negative_weights()
    test_original_reproducer()
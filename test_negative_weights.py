#!/usr/bin/env python3
"""Test script to reproduce the negative weights issue with KDE."""

import zfit
import numpy as np
import pandas as pd

# -------------------------------
class Data:
    '''
    data class
    '''
    nentries = 500
    obs = zfit.Space('x', limits=(-4, +4))

# -------------------------------
def _get_df(kind: str, scale: float, nentries: int = None) -> pd.DataFrame:
    nentries = Data.nentries if nentries is None else nentries

    arr_val = np.random.normal(loc=0, scale=1.0, size=nentries)
    arr_wgt = np.random.normal(loc=1, scale=0.1, size=nentries)

    df = pd.DataFrame({'x': arr_val, 'w': arr_wgt})
    df['kind'] = kind
    df['w'] = scale * df['w']

    return df

# -------------------------------
def main():
    df_1 = _get_df(kind='a', scale=+1.00, nentries=None)
    df_2 = _get_df(kind='b', scale=-0.01, nentries=1)

    df = pd.concat([df_1, df_2])
    df_neg = df[df.w < 0]
    print(f"Negative weights data:\n{df_neg}")

    data = zfit.data.Data.from_pandas(df=df, obs=Data.obs, weights='w')
    pdf = zfit.pdf.KDE1DimExact(data, bandwidth='isj')

    arr_x = np.linspace(-4, +4, 20)
    try:
        arr_y = pdf.pdf(arr_x).numpy()
        print(f"PDF values: {arr_y}")
    except Exception as e:
        print(f"Error: {e}")
        raise

# -------------------------------
if __name__ == '__main__':
    main()
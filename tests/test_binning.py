#  Copyright (c) 2024 zfit

import numpy as np

import zfit
from zfit.core.binning import histogramdd
import zfit.z.numpy as znp

data1 = np.random.normal(size=(1000, 3))
obs1 = zfit.Space("obs1", limits=(-100, 300))
obs2 = zfit.Space("obs2", limits=(-100, 300))
obs3 = zfit.Space("obs3", limits=(-100, 300))

obs = obs1 * obs2 * obs3


def test_histogramdd():
    histdd_kwargs = {"sample": data1}
    bincount_np, edges_np = histogramdd(**histdd_kwargs)
    bincount_true, edges_true = np.histogramdd(**histdd_kwargs)
    np.testing.assert_allclose(bincount_true, bincount_np)
    np.testing.assert_allclose(edges_true, edges_np)

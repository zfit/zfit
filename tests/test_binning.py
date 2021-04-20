#  Copyright (c) 2021 zfit

import numpy as np
import pytest

import zfit
from zfit import z
from zfit.core.binning import histogramdd

data1 = np.random.normal(size=(1000, 3))
obs1 = zfit.Space("obs1", limits=(-100, 300))
obs2 = zfit.Space("obs2", limits=(-100, 300))
obs3 = zfit.Space("obs3", limits=(-100, 300))

obs = obs1 * obs2 * obs3


def test_histogramdd():
    histdd_kwargs = {"sample": data1}
    hist = histogramdd(**histdd_kwargs)
    bincount_np, edges_np = zfit.run(hist)
    bincount_true, edges_true = np.histogramdd(**histdd_kwargs)
    np.testing.assert_allclose(bincount_true, bincount_np)
    np.testing.assert_allclose(edges_true, edges_np)


@pytest.mark.skip
def test_create_binneddata():
    data = zfit.Data.from_numpy(obs=obs, array=data1)

    histdd_kwargs = {"bins": 4}
    histdata = data.create_hist(converter=zfit.hist.histogramdd, name="histdata", bin_kwargs=histdd_kwargs)

    bincount_true, edges_true = np.histogramdd(sample=data1, **histdd_kwargs)
    bincount_np, edges_np = zfit.run(histdata.hist())
    np.testing.assert_allclose(bincount_true, bincount_np)
    np.testing.assert_allclose(edges_true, edges_np)


@pytest.mark.skip
def test_midpoints():
    edges = np.array([[-1., 0, 3, 10],
                      [-5., 0, 1, 4]])
    bincounts = np.array([[0, 0, 1],
                          [0, 5, 7],
                          [0, 3, 0],
                          [0, 0, 0]])

    edges = z.convert_to_tensor(edges)
    midpoints_true = np.array([[-0.5, 2.5],
                               [1.5, 0.5],
                               [1.5, 2.5],
                               [6.5, 0.5]])
    bincounts_nonzero, midpoints_nonzero, bincounts_nonzero_index = midpoints_from_hist(bincounts=bincounts,
                                                                                        edges=edges)
    np.testing.assert_allclose(np.array([1, 5, 7, 3]), zfit.run(bincounts_nonzero))
    np.testing.assert_allclose(midpoints_true, zfit.run(midpoints_nonzero))

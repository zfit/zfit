#  Copyright (c) 2023 zfit
import hist
import numpy as np

import zfit
import zfit.z.numpy as znp

# TODO: what is needed in this file?
from zfit import z
from zfit.core.binning import histogramdd, midpoints_from_hist, unbinned_to_binindex

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


def test_midpoints():
    edges = np.array([[-1.0, 0, 3, 10], [-5.0, 0, 1, 4]])
    bincounts = np.array([[0, 0, 1], [0, 5, 7], [0, 3, 0], [0, 0, 0]])

    edges = z.convert_to_tensor(edges)
    midpoints_true = np.array([[-0.5, 2.5], [1.5, 0.5], [1.5, 2.5], [6.5, 0.5]])
    bincounts_nonzero, midpoints_nonzero, bincounts_nonzero_index = midpoints_from_hist(
        bincounts=bincounts, edges=edges
    )
    np.testing.assert_allclose(np.array([1, 5, 7, 3]), zfit.run(bincounts_nonzero))
    np.testing.assert_allclose(midpoints_true, zfit.run(midpoints_nonzero))


def test_unbinned_to_bins():
    n = 100_000
    lower = [-5, -50, 10]
    upper = [5, 13, 100]
    values = np.random.uniform(lower, upper, size=(n, 3))
    axes = [
        hist.axis.Regular(12, lower[0], upper[0], name="x"),
        hist.axis.Regular(24, lower[1], upper[1], name="y"),
        hist.axis.Variable(
            [10, 20, 23, 26.5, 30, 35.0, 50, 60.0, 75, 78, 90.0, 100], name="z"
        ),
    ]
    h = hist.NamedHist(*axes)
    name_values = dict(zip(["x", "y", "z"], values.transpose()))
    # h.fill(**name_values)
    true_bins = h.axes.index(*name_values.values())
    space = zfit.Space(binning=axes)
    data = zfit.Data.from_tensor(space.with_binning(None), values)
    bins = znp.transpose(unbinned_to_binindex(data, space))
    np.testing.assert_array_equal(bins, true_bins)

#  Copyright (c) 2019 zfit
import numpy as np

import zfit
from zfit.core.testing import setup_function, teardown_function, tester

data1 = np.random.normal(size=(1000, 3))
obs1 = zfit.Space("obs1", limits=(-1, 3))
obs2 = zfit.Space("obs2", limits=(-1, 3))
obs3 = zfit.Space("obs3", limits=(-1, 3))

obs = obs1 * obs2 * obs3


def test_histogramdd():
    histdd_kwargs = {"sample": data1}
    hist = zfit.hist.histogramdd(**histdd_kwargs)
    bincount_np, edges_np = zfit.run(hist)
    bincount_true, edges_true = np.histogramdd(**histdd_kwargs)
    np.testing.assert_allclose(bincount_true, bincount_np)
    np.testing.assert_allclose(edges_true, edges_np)


def test_histdata():
    data = zfit.Data.from_numpy(obs=obs, array=data1)
    histdata = data.create_hist(name="histdata")

    bincount_true, edges_true = np.histogramdd(sample=data1)
    bincount_np, edges_np = zfit.run(histdata.hist())
    np.testing.assert_allclose(bincount_true, bincount_np)
    np.testing.assert_allclose(edges_true, edges_np)

#  Copyright (c) 2021 zfit
import hist
import boost_histogram as bh
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def hist1():
    from zfit._data.binneddata import BinnedData
    return BinnedData


@pytest.fixture
def holder1():
    from zfit.z import numpy as znp
    from zfit._data.binneddata import BinnedHolder
    return BinnedHolder(tf.constant('asdf'), znp.random.uniform(size=[5]), znp.random.uniform(size=[5]))


@pytest.fixture
def holder2():
    from zfit.z import numpy as znp
    from zfit._data.binneddata import BinnedHolder
    return BinnedHolder(tf.constant('asdf'), znp.random.uniform(size=[5]), znp.random.uniform(size=[5]))


def test_composite(holder1, holder2):
    from zfit.z import numpy as znp

    count = 0

    @tf.function
    def tot_variances(x):
        nonlocal count
        count += 1
        return znp.sum(x.variances)

    trueval = np.sum(holder1.variances)
    testval = tot_variances(holder1)
    assert pytest.approx(trueval, testval)
    actual_count = count
    _ = tot_variances(holder2)
    assert count == actual_count


def test_from_and_to_hist():
    h3 = hist.NamedHist(
        hist.axis.Regular(20, -3, 3, name="x", flow=False), hist.axis.Regular(20, -3, 3, name="y", flow=False),
        storage=hist.storage.Weight()
    )

    x2 = np.random.randn(1_000)
    y2 = 0.5 * np.random.randn(1_000)

    h3.fill(x=x2, y=y2)

    from zfit._data.binneddata import BinnedData
    for _ in range(100):  # make sure this works many times
        h1 = BinnedData.from_hist(h3)
        np.testing.assert_allclose(h1.variances(), h3.variances())
        np.testing.assert_allclose(h1.values(), h3.values())

        h3recreated = h1.to_hist()
        assert h3recreated == h3

    bh3 = bh.Histogram(h1)
    np.testing.assert_allclose(h1.variances(), bh3.variances())
    np.testing.assert_allclose(h1.values(), bh3.values())


def test_values():
    assert True


def test_variance():
    assert True


def test_counts():
    assert True
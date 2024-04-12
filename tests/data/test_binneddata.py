#  Copyright (c) 2024 zfit
import boost_histogram as bh
import hist
import numpy as np
import pytest
import tensorflow as tf

import zfit


@pytest.fixture
def hist1():
    from zfit._data.binneddatav1 import BinnedData

    return BinnedData


@pytest.fixture
def holder1():
    from zfit.z import numpy as znp
    from zfit._data.binneddatav1 import BinnedHolder

    return BinnedHolder(
        tf.constant("asdf"), znp.random.uniform(size=[5]), znp.random.uniform(size=[5])
    )


@pytest.fixture
def holder2():
    from zfit.z import numpy as znp
    from zfit._data.binneddatav1 import BinnedHolder

    return BinnedHolder(
        tf.constant("asdf"), znp.random.uniform(size=[5]), znp.random.uniform(size=[5])
    )


@pytest.mark.skip(
    "Currently not a composite tensor, test when we can pass it through again"
)
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


def test_from_and_to_binned():
    h3 = hist.Hist(
        hist.axis.Regular(3, -3, 3, name="x", flow=False),
        hist.axis.Regular(2, -5, 5, name="y", flow=False),
        storage=hist.storage.Weight(),
    )

    x2 = np.random.randn(1_000)
    y2 = 0.5 * np.random.randn(1_000)

    h3.fill(x=x2, y=y2)

    from zfit._data.binneddatav1 import BinnedData

    h1 = BinnedData.from_hist(h3)
    for _ in range(10):  # make sure this works many times
        unbinned = h1.to_unbinned()
        binned = unbinned.to_binned(space=h1.space)
        np.testing.assert_allclose(binned.values(), h1.values())
        # we can't test the variances, this info is lost
        h1 = binned
    bh3 = bh.Histogram(h1)
    np.testing.assert_allclose(h1.values(), bh3.values())


def test_from_and_to_hist():
    h3 = hist.NamedHist(
        hist.axis.Regular(25, -3.5, 3, name="x", flow=False),
        hist.axis.Regular(21, -4, 5, name="y", flow=False),
        storage=hist.storage.Weight(),
    )

    x2 = np.random.randn(1_000)
    y2 = 0.5 * np.random.randn(1_000)

    h3.fill(x=x2, y=y2)

    from zfit._data.binneddatav1 import BinnedData

    for _ in range(10):  # make sure this works many times
        h1 = BinnedData.from_hist(h3)
        np.testing.assert_allclose(h1.variances(), h3.variances())
        np.testing.assert_allclose(h1.values(), h3.values())
        unbinned = h1.to_unbinned()
        assert unbinned.value().shape[1] == 2
        assert unbinned.value().shape[0] == unbinned.weights.shape[0]

        h3recreated = h1.to_hist()
        assert h3recreated == h3

    bh3 = bh.Histogram(h1)
    np.testing.assert_allclose(h1.variances(), bh3.variances())
    np.testing.assert_allclose(h1.values(), bh3.values())

def test_binned_data_from_unbinned():
    import zfit

    axis1 = hist.axis.Regular(25, -3.5, 3, name="x", flow=False)
    axis2 = hist.axis.Regular(21, -4, 5, name="y", flow=False)
    h3 = hist.NamedHist(
        axis1,
        axis2,
        storage=hist.storage.Weight(),
    )

    x2 = np.random.randn(1_000)
    y2 = 0.5 * np.random.randn(1_000)

    h3.fill(x=x2, y=y2)

    from zfit._data.binneddatav1 import BinnedData

    xobs = zfit.Space("x", binning=axis1)
    yobs = zfit.Space("y", binning=axis2)
    obsbinned = xobs * yobs
    obs_unbinned = xobs.with_binning(False) * yobs.with_binning(False)
    array = np.stack([x2, y2], axis=1)
    data = zfit.Data.from_numpy(obs=obs_unbinned, array=array)
    binned_data_init = zfit.Data.from_numpy(obs=obsbinned, array=array)
    binned_data = BinnedData.from_unbinned(data=data, space=obsbinned)
    np.testing.assert_allclose(binned_data.values(), h3.values())
    np.testing.assert_allclose(binned_data.variances(), h3.variances())
    np.testing.assert_allclose(binned_data_init.values(), h3.values())
    np.testing.assert_allclose(binned_data_init.variances(), h3.variances())




def test_with_obs():
    from zfit._data.binneddatav1 import BinnedData

    h1 = hist.NamedHist(
        hist.axis.Regular(25, -3.5, 3, name="x", flow=False),
        hist.axis.Regular(21, -4, 5, name="y", flow=False),
        hist.axis.Regular(15, -2, 1, name="z", flow=False),
        storage=hist.storage.Weight(),
    )

    x2 = np.random.randn(1_000)
    y2 = 0.5 * np.random.randn(1_000)
    z2 = 0.3 * np.random.randn(1_000)

    h1.fill(x=x2, y=y2, z=z2)
    h = BinnedData.from_hist(h1)
    obs = ("x", "y", "z")
    obs2 = ("y", "x", "z")
    assert obs == h.obs
    h2 = h.with_obs(obs2)
    assert h2.obs == obs2
    np.testing.assert_allclose(h.values()[:, 3, 5], h2.values()[3, :, 5])
    np.testing.assert_allclose(h.variances()[:, 3, 5], h2.variances()[3, :, 5])


def test_valid_input():
    from zfit._data.binneddatav1 import BinnedHolder
    import zfit.z.numpy as znp
    import zfit
    from zfit.exception import ShapeIncompatibleError

    obs10 = zfit.Space("x", binning=zfit.binned.RegularBinning(10, -3, 4, name="x"))
    obs5 = zfit.Space("x", binning=zfit.binned.RegularBinning(5, -3, 4, name="x"))
    _ = BinnedHolder(
        obs10, znp.random.uniform(size=[10]), znp.random.uniform(size=[10])
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = BinnedHolder(
            obs10, znp.random.uniform(size=[5]), znp.random.uniform(size=[5])
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = BinnedHolder(
            obs5, znp.random.uniform(size=[10]), znp.random.uniform(size=[5])
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = BinnedHolder(
            obs5, znp.random.uniform(size=[10]), znp.random.uniform(size=[10])
        )
    _ = BinnedHolder(obs5, znp.random.uniform(size=[5]), None)
    _ = BinnedHolder(obs10, znp.random.uniform(size=[10]), None)
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = BinnedHolder(obs5, znp.random.uniform(size=[10]), None)
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = BinnedHolder(obs10, znp.random.uniform(size=[5]), None)
    with pytest.raises(ShapeIncompatibleError):
        _ = BinnedHolder(
            obs10, znp.random.uniform(size=[10]), znp.random.uniform(size=[10, 2])
        )
    with pytest.raises(ShapeIncompatibleError):
        _ = BinnedHolder(
            obs10, znp.random.uniform(size=[10, 2]), znp.random.uniform(size=[10, 2])
        )
    with pytest.raises(ShapeIncompatibleError):
        _ = BinnedHolder(obs10, znp.random.uniform(size=[5, 2]), None)


def test_variance():
    import zfit
    import zfit.z.numpy as znp

    binning1 = zfit.binned.RegularBinning(3, -3.5, 3, name="x")
    obs = zfit.Space("x", binning=binning1)
    values = znp.array([100.0, 200, 50])
    data = zfit.data.BinnedData.from_tensor(obs, values=values, variances=True)
    data2 = zfit.data.BinnedData.from_tensor(obs, values=values, variances=values**0.5)
    np.testing.assert_allclose(data.variances(), data2.variances())


def test_binneddata_with_variance_method():
    bins1 = 51
    bins2 = 55
    bins3 = 64
    size = (bins1, bins2, bins3)
    sample = np.random.uniform(100, 10000, size=size)
    variance = np.random.uniform(100, 10000, size=size)
    variance2 = np.random.uniform(100, 10000, size=size)

    space1 = zfit.Space('obs1', limits=(-100, 100), binning=bins1)
    space2 = zfit.Space('obs2', limits=(-200, 100), binning=bins2)
    space3 = zfit.Space('obs3', limits=(-150, 350), binning=bins3)
    obs = space1 * space2 * space3

    data = zfit.data.BinnedData.from_tensor(space=obs, values=sample, variances=variance)
    assert data.variances() is not None
    np.testing.assert_allclose(data.variances(), variance)
    data2 = data.with_variances(variance2)
    np.testing.assert_allclose(data2.variances(), variance2)
    data3 = data2.with_variances(None)
    assert data3.variances() is None
    data4 = data3.with_variances(variance)
    np.testing.assert_allclose(data4.variances(), variance)
    data4obs = data4.with_obs(obs=space2 * space1 * space3)
    assert data4obs.space == space2 * space1 * space3
    np.testing.assert_allclose(data4obs.values(), np.moveaxis(sample, [0, 1, 2], [1, 0, 2]))

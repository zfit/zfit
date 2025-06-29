#  Copyright (c) 2025 zfit
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

    num_entries = 1_000
    x2 = np.random.randn(num_entries)
    y2 = 0.5 * np.random.randn(num_entries)

    h3.fill(x=x2, y=y2)

    from zfit.data import BinnedData

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
    assert pytest.approx(data.samplesize) == binned_data_init.samplesize




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


# refers to issue https://github.com/zfit/zfit/issues/602
def test_binned_from_numpy_issue602():
    import zfit
    import numpy as np

    low, high = -5, 10
    binning = zfit.binned.RegularBinning(15, low, high, name="obs") # Define the binning for the observable
    obs = zfit.Space(obs='obs', binning=binning)                    # Create the observable space

    size_normal = 10_000                                            # Set the size of the normal distribution data
    data_normal_np = np.random.normal(size=size_normal, scale=2)    # Generate normal distribution data

    data_normal = zfit.Data(data_normal_np, obs=obs)                # Create a zfit Data object with the normal distribution data
    assert data_normal_np.shape[0] >= data_normal.num_entries           # Check that the number of events in the data is correct


def test_binned_data_array_protocol():
    """Test the numpy __array__ protocol implementation for BinnedData objects."""
    import zfit

    # Create test data
    binning = zfit.binned.RegularBinning(10, -3, 3, name="x")
    obs = zfit.Space("x", binning=binning)

    # Create test values (histogram counts)
    values = np.array([5, 10, 15, 20, 25, 30, 25, 20, 15, 10])
    variances = np.sqrt(values)  # Poisson variances

    # Create BinnedData
    binned_data = zfit.data.BinnedData.from_tensor(
        space=obs, values=values, variances=variances
    )

    # Test basic __array__ conversion
    arr = np.array(binned_data)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, values)
    assert arr.shape == (10,)

    # Test asarray conversion
    arr2 = np.asarray(binned_data)
    assert isinstance(arr2, np.ndarray)
    np.testing.assert_array_equal(arr2, values)

    # Test dtype conversion
    arr_float32 = np.array(binned_data, dtype=np.float32)
    assert arr_float32.dtype == np.float32
    np.testing.assert_array_almost_equal(arr_float32, values.astype(np.float32))

    # Test copy=True
    arr_copy = np.array(binned_data, copy=True)
    assert isinstance(arr_copy, np.ndarray)
    np.testing.assert_array_equal(arr_copy, values)
    # Modify copy to ensure it's independent
    arr_copy[0] = 999
    assert not np.array_equal(arr_copy, values)

    # Test copy=False with same dtype (should not raise)
    arr_nocopy = np.array(binned_data, copy=False)
    assert isinstance(arr_nocopy, np.ndarray)
    np.testing.assert_array_equal(arr_nocopy, values)


def test_binned_data_array_protocol_numpy_functions():
    """Test that BinnedData objects work with various numpy functions."""
    import zfit

    # Create test data (2D histogram)
    binning_x = zfit.binned.RegularBinning(5, -2, 2, name="x")
    binning_y = zfit.binned.RegularBinning(4, -1, 1, name="y")
    space_x = zfit.Space("x", binning=binning_x)
    space_y = zfit.Space("y", binning=binning_y)
    obs = space_x * space_y

    # Create 2D histogram values
    values = np.random.poisson(10, size=(5, 4))
    binned_data = zfit.data.BinnedData.from_tensor(space=obs, values=values)

    # Test with np.mean
    mean_data = np.mean(binned_data)
    mean_values = np.mean(values)
    np.testing.assert_almost_equal(mean_data, mean_values)

    # Test with np.sum
    sum_data = np.sum(binned_data)
    sum_values = np.sum(values)
    np.testing.assert_almost_equal(sum_data, sum_values)

    # Test with np.max and np.min
    max_data = np.max(binned_data)
    max_values = np.max(values)
    np.testing.assert_equal(max_data, max_values)

    min_data = np.min(binned_data)
    min_values = np.min(values)
    np.testing.assert_equal(min_data, min_values)

    # Test array operations using numpy on the array representation
    arr_data = np.array(binned_data)
    scaled_data = arr_data * 2
    scaled_values = values * 2
    np.testing.assert_array_equal(scaled_data, scaled_values)


def test_binned_data_array_protocol_1d():
    """Test __array__ protocol with 1D BinnedData."""
    import zfit

    # Create 1D binned data
    binning = zfit.binned.RegularBinning(8, -4, 4, name="x")
    obs = zfit.Space("x", binning=binning)

    values = np.array([1, 3, 7, 12, 15, 12, 7, 3])
    binned_data = zfit.data.BinnedData.from_tensor(space=obs, values=values)

    # Test array conversion
    arr = np.array(binned_data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8,)
    np.testing.assert_array_equal(arr, values)

    # Test that it works with numpy functions
    total = np.sum(arr)
    expected_total = np.sum(values)
    np.testing.assert_equal(total, expected_total)


def test_binned_data_array_protocol_from_hist():
    """Test __array__ protocol with BinnedData created from hist object."""
    import zfit
    import hist

    # Create hist object
    h = hist.Hist(
        hist.axis.Regular(6, -3, 3, name="x", flow=False),
        storage=hist.storage.Weight(),
    )

    # Fill with some data
    data = np.random.normal(0, 1, 1000)
    h.fill(x=data)

    # Convert to BinnedData
    binned_data = zfit.data.BinnedData.from_hist(h)

    # Test __array__ method (should inherit)
    arr = np.array(binned_data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)

    # Test that values match
    np.testing.assert_array_equal(arr, h.values())

    # Test with numpy functions
    mean_result = np.mean(arr)
    assert isinstance(mean_result, (float, np.floating))


def test_binned_data_array_protocol_comparison():
    """Test that __array__ returns the same as values() method."""
    import zfit

    # Create test data
    binning = zfit.binned.RegularBinning(7, -1, 1, name="x")
    obs = zfit.Space("x", binning=binning)

    values = np.array([2, 5, 8, 12, 8, 5, 2])
    binned_data = zfit.data.BinnedData.from_tensor(space=obs, values=values)

    # Test that __array__ returns the same as values()
    arr_protocol = np.array(binned_data)
    arr_values = binned_data.values().numpy()

    np.testing.assert_array_equal(arr_protocol, arr_values)

    # Test that both work the same with numpy functions
    sum_protocol = np.sum(binned_data)
    sum_values = np.sum(binned_data.values())

    np.testing.assert_equal(sum_protocol, sum_values)

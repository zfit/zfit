#  Copyright (c) 2024 zfit
import pickle

import numpy as np
import pytest
import tensorflow as tf

import zfit._data.unbinneddata
import zfit._variables.axis as axis
import zfit.z.numpy as znp
from zfit.util.exception import ObsIncompatibleError


@pytest.mark.skip
def test_basic():
    n = 1000
    nobs = 3
    axes = axis.SpaceV2(
        [axis.UnbinnedAxis("x"), axis.UnbinnedAxis("y"), axis.UnbinnedAxis("z")]
    )
    tensor = znp.random.uniform(size=(n, nobs))
    weights = znp.random.uniform(size=(n,))
    data = zfit._data.unbinneddata.UnbinnedData(data=tensor, axes=axes, weights=weights)

    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)
    # assert data.axes == unpickled.axes
    np.testing.assert_array_equal(data.weights, unpickled.weights)
    np.testing.assert_array_equal(data.data, unpickled.data)

    array = data["y"]
    np.testing.assert_array_equal(array, tensor[:, 1])

    # af = asdf.AsdfFile(data.obj_to_repr())
    # with io.BytesIO(b'file') as f:
    #     af.write_to(f)


def test_data_conversion():
    obs1 = zfit.Space("obs1", (-1, 100))
    obs2 = zfit.Space("obs2", (-1, 100))
    obs12 = obs1 * obs2
    testdata1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    from zfit.core.data import convert_to_data

    out1 = convert_to_data(testdata1, obs=obs1)
    assert out1.value().shape == (5, 1)
    np.testing.assert_allclose(out1["obs1"], testdata1)
    # check that it fails without obs
    with pytest.raises(ValueError):
        convert_to_data(testdata1)

    testdata2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 22.0, 34.0, 41.0, 5.0]]).T
    out2 = convert_to_data(testdata2, obs=obs12)
    assert out2.value().shape == (5, 2)
    np.testing.assert_allclose(out2["obs1"], testdata2[:, 0])
    np.testing.assert_allclose(out2["obs2"], testdata2[:, 1])
    with pytest.raises(ValueError):
        convert_to_data(testdata2)

    # test dataframe
    import pandas as pd

    df = pd.DataFrame(testdata2, columns=["obs1", "obs2"])
    out3 = convert_to_data(df)
    assert out3.value().shape == (5, 2)
    np.testing.assert_allclose(out3["obs1"], testdata2[:, 0])
    np.testing.assert_allclose(out3["obs2"], testdata2[:, 1])

    # test single value
    single_value = 5.0
    out4 = convert_to_data(single_value, obs=obs1)
    assert out4.value().shape == (1, 1)
    np.testing.assert_allclose(out4["obs1"], single_value)
    with pytest.raises(ValueError):
        convert_to_data(single_value)

    # test single int
    single_value = 5
    out5 = convert_to_data(single_value, obs=obs1)
    assert out5.value().shape == (1, 1)
    np.testing.assert_allclose(out5["obs1"], single_value)
    with pytest.raises(ValueError):
        convert_to_data(single_value)

def test_unbinneddata_getitem_index():
    n = 1000
    nobs = 3
    space = zfit.Space(obs="obs1", axes=(0,), limits=(-1, 1)) * zfit.Space(obs="obs2", axes=(1,), limits=(-1, 1)) * zfit.Space(obs="obs3", axes=(2,), limits=(-1, 1))
    space = space.with_autofill_axes()
    arr = znp.random.uniform(size=(n, nobs))
    weights = znp.random.uniform(size=(n,))
    data = zfit.Data.from_numpy(obs=space, array=arr, weights=weights)
    assert space.axes == (0, 1, 2)

    data0 = data.with_obs(0)
    data1 = data.with_obs(1)

    assert data0.value().shape == (n, 1)
    assert data1.value().shape == (n, 1)
    np.testing.assert_array_equal(data0.value(), arr[:, 0:1])
    np.testing.assert_array_equal(data1.value(), arr[:, 1:2])
    np.testing.assert_array_equal(data0["obs1"], arr[:, 0])
    np.testing.assert_array_equal(data1["obs2"], arr[:, 1])
    np.testing.assert_array_equal(data["obs1"], arr[:, 0])
    np.testing.assert_array_equal(data["obs2"], arr[:, 1])

    data10 = data.with_obs([1, 0])
    assert data10.value().shape == (n, 2)
    np.testing.assert_array_equal(data10.value(), arr[:, 1::-1])
    np.testing.assert_array_equal(data10["obs2"], arr[:, 1])
    np.testing.assert_array_equal(data10["obs1"], arr[:, 0])

    np.testing.assert_array_equal(data10[0], arr[:, 1])
    np.testing.assert_array_equal(data10[1], arr[:, 0])




def test_unbinned_data_concat_index():
    n = 1000
    n2 = 552
    n3 = 123
    nobs = 3
    space1 = zfit.Space(obs="obs1", limits=(-1, 1))
    space2 = zfit.Space(obs="obs2", limits=(-11, 5))
    space3 = zfit.Space(obs="obs3", limits=(-15, 3))
    space = space1 * space2 * space3

    arr1 = znp.random.uniform(size=(n, nobs))
    weights1 = znp.random.uniform(size=(n,))
    data1w = zfit.Data.from_numpy(obs=space, array=arr1, weights=weights1)
    data1now = zfit.Data.from_numpy(obs=space, array=arr1)

    arr2 = znp.random.uniform(size=(n2, nobs))
    weights2 = znp.random.uniform(size=(n2,))
    data2w = zfit.Data.from_numpy(obs=space, array=arr2, weights=weights2)
    data2now = zfit.Data.from_numpy(obs=space, array=arr2)
    # todo: if we want to switch argument orders, legacy element
    # with pytest.warns(UserWarning):
    #     data2now_order = zfit.Data.from_numpy(space, arr2)  # wrong order, legacy check
    # np.testing.assert_array_equal(data2now_order.value(), data2w.value())
    # assert data2now_order.space == data2w.space


    arr3 = znp.random.uniform(size=(n3, nobs))
    weights3 = znp.random.uniform(size=(n3,))
    data3w = zfit.Data.from_numpy(obs=space, array=arr3, weights=weights3)
    data3now = zfit.Data.from_numpy(obs=space, array=arr3)

    dataw = zfit.data.concat([data1w, data2w])
    np.testing.assert_array_equal(dataw.value(), np.concatenate([arr1, arr2]))
    assert dataw.space == space
    np.testing.assert_array_equal(dataw.weights, np.concatenate([weights1, weights2]))

    datawall = zfit.data.concat([data1w, data2w, data3w])
    np.testing.assert_array_equal(datawall.value(), np.concatenate([arr1, arr2, arr3]))
    assert datawall.space == space
    np.testing.assert_array_equal(datawall.weights, np.concatenate([weights1, weights2, weights3]))
    assert datawall.space == space
    spacemixed = space1 * space3 * space2

    mixedarray = np.concatenate([np.array(arr1)[:, [0, 2, 1]], np.array(arr2)[:, [0, 2, 1]], np.array(arr3)[:, [0, 2, 1]],], axis=0)

    datawallmixed = zfit.data.concat([data1w, data2w, data3w], obs=spacemixed)
    np.testing.assert_array_equal(datawallmixed.value(), mixedarray)
    assert datawallmixed.space == spacemixed
    np.testing.assert_array_equal(datawallmixed.weights, np.concatenate([weights1, weights2, weights3]))

    datanow = zfit.data.concat([data1now, data2now])
    np.testing.assert_array_equal(datanow.value(), np.concatenate([arr1, arr2]))
    assert datanow.space == space
    assert datanow.weights is None

    datamixedw = zfit.data.concat([data1w, data2now, data3w])
    np.testing.assert_array_equal(datamixedw.value(), np.concatenate([arr1, arr2, arr3]))
    assert datamixedw.space == space
    np.testing.assert_array_equal(datamixedw.weights, np.concatenate([weights1, np.ones_like(weights2), weights3]))

def test_unbinned_data_concat_obs():
    n = 1000
    nobs = 1
    space1 = zfit.Space(obs="obs1", limits=(-1, 1))
    space2 = zfit.Space(obs="obs2", limits=(-11, 5))
    space3 = zfit.Space(obs="obs3", limits=(-15, 3))
    space = space1 * space2 * space3

    arr1 = znp.random.uniform(size=(n, nobs))
    weights1 = znp.random.uniform(size=(n,))
    data1w = zfit.Data.from_numpy(obs=space1, array=arr1, weights=weights1)
    data1now = zfit.Data.from_numpy(obs=space1, array=arr1)

    arr2 = znp.random.uniform(size=(n, nobs))
    weights2 = znp.random.uniform(size=(n,))
    data2w = zfit.Data.from_numpy(obs=space2, array=arr2, weights=weights2)
    data2now = zfit.Data.from_numpy(obs=space2, array=arr2)

    arr3 = znp.random.uniform(size=(n, nobs))
    weights3 = znp.random.uniform(size=(n,))
    data3w = zfit.Data.from_numpy(obs=space3, array=arr3, weights=weights3)
    data3now = zfit.Data.from_numpy(obs=space3, array=arr3)

    data12w = zfit.data.concat([data1w, data2w], axis=1)
    np.testing.assert_array_equal(data12w.value(), np.concatenate([arr1, arr2], axis=1))
    assert data12w.space == space1 * space2
    np.testing.assert_array_equal(data12w.weights, weights1 * weights2)

    data123w = zfit.data.concat([data1w, data2w, data3w], axis=1)
    np.testing.assert_array_equal(data123w.value(), np.concatenate([arr1, arr2, arr3], axis=1))
    assert data123w.space == space
    np.testing.assert_array_equal(data123w.weights, weights1 * weights2 * weights3)

    spacemixed = space1 * space3 * space2
    data123mixed = zfit.data.concat([data1w, data2w, data3w], obs=spacemixed, axis=1)
    mixedarray = np.concatenate([arr1, arr3, arr2], axis=1)
    assert data123mixed.space == spacemixed
    np.testing.assert_array_equal(data123mixed.value(), mixedarray)
    np.testing.assert_allclose(data123mixed.weights, weights1 * weights3 * weights2)

    datanow = zfit.data.concat([data1now, data2now], axis=1)
    np.testing.assert_array_equal(datanow.value(), np.concatenate([arr1, arr2], axis=1))
    assert datanow.space == space1 * space2
    assert datanow.weights is None

    datamixedw = zfit.data.concat([data1w, data2now, data3w], axis=1)
    np.testing.assert_array_equal(datamixedw.value(), np.concatenate([arr1, arr2, arr3], axis=1))
    assert datamixedw.space == space
    np.testing.assert_allclose(datamixedw.weights, weights1 * weights3)

def test_unbinned_data_concat_obs_errors():

    space1 = zfit.Space(obs="obs1", limits=(-1, 1))
    space2 = zfit.Space(obs="obs2", limits=(-11, 5))
    space3 = zfit.Space(obs="obs3", limits=(-15, 3))
    space12 = space1 * space2

    arr1 = znp.random.uniform(size=(1000, 1))
    weights1 = znp.random.uniform(size=(1000,))
    data1w = zfit.Data.from_numpy(obs=space1, array=arr1, weights=weights1)

    arr2lessevents = znp.random.uniform(size=(10, 1))
    weights2 = znp.random.uniform(size=(10,))
    data2wless = zfit.Data.from_numpy(obs=space2, array=arr2lessevents, weights=weights2)

    arr2 = znp.random.uniform(size=(1000, 1))
    weights2 = znp.random.uniform(size=(1000,))
    data2w = zfit.Data.from_numpy(obs=space2, array=arr2, weights=weights2)

    arr3 = znp.random.uniform(size=(1000, 1))
    weights3 = znp.random.uniform(size=(1000,))
    data3w = zfit.Data.from_numpy(obs=space3, array=arr3, weights=weights3)



    with pytest.raises(tf.errors.InvalidArgumentError):
        zfit.data.concat([data1w, data2wless], axis=1)

    arr12 = znp.random.uniform(size=(1000, 2))
    weights12 = znp.random.uniform(size=(1000,))
    data12w = zfit.Data.from_numpy(obs=space12, array=arr12, weights=weights12)

    with pytest.raises(ObsIncompatibleError):
        zfit.data.concat([data1w, data12w], axis=1)

    with pytest.raises(ObsIncompatibleError):
        zfit.data.concat([data1w, data3w], obs=space12, axis=1)


def test_data_array_protocol():
    """Test the numpy __array__ protocol implementation for Data objects."""
    import zfit

    # Test data setup
    n = 100
    nobs = 2
    space_x = zfit.Space("x", limits=(-1, 1))
    space_y = zfit.Space("y", limits=(-2, 2))
    obs = space_x * space_y

    # Create test data
    np_data = np.random.uniform(-1, 1, size=(n, nobs))
    weights = np.random.uniform(0.5, 1.5, size=(n,))

    # Test unweighted data
    data = zfit.Data.from_numpy(obs=obs, array=np_data)

    # Test basic __array__ conversion
    arr = np.array(data)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np_data)
    assert arr.shape == (n, nobs)

    # Test asarray conversion
    arr2 = np.asarray(data)
    assert isinstance(arr2, np.ndarray)
    np.testing.assert_array_equal(arr2, np_data)

    # Test dtype conversion
    arr_float32 = np.array(data, dtype=np.float32)
    assert arr_float32.dtype == np.float32
    np.testing.assert_array_almost_equal(arr_float32, np_data.astype(np.float32))

    # Test copy=True
    arr_copy = np.array(data, copy=True)
    assert isinstance(arr_copy, np.ndarray)
    np.testing.assert_array_equal(arr_copy, np_data)
    # Modify copy to ensure it's independent
    arr_copy[0, 0] = 999
    assert not np.array_equal(arr_copy, np_data)

    # Test copy=False with same dtype (should not raise)
    arr_nocopy = np.array(data, copy=False)
    assert isinstance(arr_nocopy, np.ndarray)
    np.testing.assert_array_equal(arr_nocopy, np_data)

    # Note: NumPy may handle dtype conversion with copy=False internally,
    # so we don't test for the exception here as the behavior can vary


def test_data_array_protocol_weighted():
    """Test the numpy __array__ protocol with weighted data."""
    import zfit

    # Test data setup
    n = 50
    obs = zfit.Space("x", limits=(-1, 1))

    # Create test data with weights
    np_data = np.random.uniform(-1, 1, size=(n, 1))
    weights = np.random.uniform(0.5, 1.5, size=(n,))

    data_weighted = zfit.Data.from_numpy(obs=obs, array=np_data, weights=weights)

    # Test that __array__ returns the data, not the weights
    arr = np.array(data_weighted)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np_data)
    assert arr.shape == (n, 1)

    # Verify weights are separate (not included in array conversion)
    assert data_weighted.has_weights
    np.testing.assert_array_equal(data_weighted.weights, weights)


def test_data_array_protocol_numpy_functions():
    """Test that Data objects work with various numpy functions."""
    import zfit

    # Test data setup
    n = 50
    nobs = 2
    space_x = zfit.Space("x", limits=(-1, 1))
    space_y = zfit.Space("y", limits=(-2, 2))
    obs = space_x * space_y

    # Create test data - make sure it's within the space limits
    np_data = np.empty((n, nobs))
    np_data[:, 0] = np.random.uniform(-0.9, 0.9, size=n)  # x: within (-1, 1)
    np_data[:, 1] = np.random.uniform(-1.9, 1.9, size=n)  # y: within (-2, 2)
    data = zfit.Data.from_numpy(obs=obs, array=np_data)

    # Test with np.mean
    mean_data = np.mean(data, axis=0)
    mean_np = np.mean(np_data, axis=0)
    np.testing.assert_array_almost_equal(mean_data, mean_np)

    # Test with np.std
    std_data = np.std(data, axis=0)
    std_np = np.std(np_data, axis=0)
    np.testing.assert_array_almost_equal(std_data, std_np)

    # Test with np.sum
    sum_data = np.sum(data, axis=0)
    sum_np = np.sum(np_data, axis=0)
    np.testing.assert_array_almost_equal(sum_data, sum_np)

    # Test with np.concatenate
    # Create data2 with guarantee_limits to avoid cutting
    data2 = zfit.Data.from_numpy(obs=obs, array=np_data * 0.5, guarantee_limits=True)  # Scale down to ensure within limits
    concat_result = np.concatenate([data, data2], axis=0)
    expected_concat = np.concatenate([np_data, np_data * 0.5], axis=0)
    np.testing.assert_array_equal(concat_result, expected_concat)

    # Test with np.reshape
    if n >= 10:
        reshaped = np.reshape(data, (n * nobs,))
        expected_reshaped = np.reshape(np_data, (n * nobs,))
        np.testing.assert_array_equal(reshaped, expected_reshaped)


def test_data_array_protocol_single_obs():
    """Test __array__ protocol with single observable data."""
    import zfit

    # Test data setup
    n = 30
    obs = zfit.Space("x", limits=(-1, 1))

    # Create test data (1D)
    np_data_1d = np.random.uniform(-1, 1, size=(n,))
    data = zfit.Data.from_numpy(obs=obs, array=np_data_1d)

    # Test array conversion
    arr = np.array(data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (n, 1)  # zfit should expand to 2D
    np.testing.assert_array_equal(arr.squeeze(), np_data_1d)

    # Test that it works with numpy functions
    mean_result = np.mean(arr)
    expected_mean = np.mean(np_data_1d)
    np.testing.assert_almost_equal(mean_result, expected_mean)


def test_sampler_data_array_protocol():
    """Test the numpy __array__ protocol with SamplerData objects."""
    import zfit

    # Test data setup
    n = 50
    obs = zfit.Space("x", limits=(-1, 1))

    def sample_func(n_samples, params):
        # Generate samples within the limits (-1, 1)
        import tensorflow as tf
        from zfit import ztypes
        return tf.random.uniform((n_samples, 1), minval=-0.9, maxval=0.9, dtype=ztypes.float)

    # Create SamplerData
    sampler_data = zfit.data.SamplerData.from_sampler(
        sample_func=sample_func,
        n=n,
        obs=obs
    )

    # Test that __array__ works (should inherit from Data)
    arr = np.array(sampler_data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (n, 1)

    # Test with numpy functions
    mean_result = np.mean(arr)
    assert isinstance(mean_result, (float, np.floating))

    # Test after resampling
    sampler_data.resample()
    arr2 = np.array(sampler_data)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == (n, 1)
    # Arrays should be different after resampling
    assert not np.array_equal(arr, arr2)

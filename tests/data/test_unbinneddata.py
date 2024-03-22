#  Copyright (c) 2024 zfit
import pickle

import numpy as np
import pytest

import zfit._data.unbinneddata
import zfit._variables.axis as axis
import zfit.z.numpy as znp


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

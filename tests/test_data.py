#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester


import copy

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot

import zfit

from zfit.core.testing import setup_function, teardown_function, tester

obs1 = ('obs1', 'obs2', 'obs3')

example_data1 = np.random.random(size=(7, len(obs1)))


def create_data1():
    return zfit.Data.from_numpy(obs=obs1, array=example_data1)


# def test_from_root_iter():
#     from skhep_testdata import data_path
#
#     path_root = data_path("uproot-Zmumu.root")
#
#     branches = ['pt1', 'pt2']
#
#     data = zfit.Data.from_root(path=path_root, treepath='events', branches=branches)
#
#     x = data.value()


@pytest.mark.parametrize("weights_factory", [lambda: None,
                                             lambda: 2. * tf.ones(shape=(1000,), dtype=tf.float64),
                                             lambda: np.random.normal(size=1000),
                                             lambda: 'eta1'])
def test_from_root(weights_factory):
    weights = weights_factory()

    from skhep_testdata import data_path

    path_root = data_path("uproot-Zmumu.root")

    branches = ['pt1', 'pt2', "phi2"]
    f = uproot.open(path_root)
    tree = f['events']

    true_data = tree.pandas.df()

    data = zfit.Data.from_root(path=path_root, treepath='events', branches=branches, weights=weights)
    x = data.value()
    x_np = zfit.run(x)
    if weights is not None:
        weights_np = zfit.run(data.weights)
    else:
        weights_np = weights
    np.testing.assert_allclose(x_np, true_data[branches].as_matrix())
    if weights is not None:
        true_weights = weights if not isinstance(weights, str) else true_data[weights].as_matrix()
        if isinstance(true_weights, tf.Tensor):
            true_weights = zfit.run(true_weights)
        np.testing.assert_allclose(weights_np, true_weights)
    else:
        assert weights_np is None


@pytest.mark.parametrize("weights_factory", [lambda: None,
                                             lambda: 2. * tf.ones(shape=(1000,), dtype=tf.float64),
                                             lambda: np.random.normal(size=1000), ])
def test_from_numpy(weights_factory):
    weights = weights_factory()

    example_data = np.random.random(size=(1000, len(obs1)))
    data = zfit.Data.from_numpy(obs=obs1, array=example_data, weights=weights)
    x = data.value()
    weights_from_data = data.weights
    if weights_from_data is not None:
        weights_from_data = zfit.run(weights_from_data)
    x_np = zfit.run(x)
    np.testing.assert_array_equal(example_data, x_np)
    if isinstance(weights, tf.Tensor):
        weights = zfit.run(weights)
    if weights is not None:
        np.testing.assert_allclose(weights_from_data, weights)
    else:
        assert weights_from_data is None


def test_from_to_pandas():
    dtype = np.float32
    example_data_np = np.random.random(size=(1000, len(obs1)))
    example_data = pd.DataFrame(data=example_data_np, columns=obs1)
    data = zfit.Data.from_pandas(obs=obs1, df=example_data, dtype=dtype)
    x = data.value()
    assert x.dtype == dtype
    x_np = zfit.run(x)
    assert x_np.dtype == dtype
    np.testing.assert_array_equal(example_data_np.astype(dtype=dtype), x_np)

    # test auto obs retreavel
    example_data2 = pd.DataFrame(data=example_data_np, columns=obs1)
    data2 = zfit.Data.from_pandas(df=example_data2)
    assert data2.obs == obs1
    x2 = data2.value()
    x_np2 = zfit.run(x2)
    np.testing.assert_array_equal(example_data_np, x_np2)

    df = data2.to_pandas()
    assert all(df == example_data)


@pytest.mark.parametrize("weights_factory", [lambda: None,
                                             lambda: 2. * tf.ones(shape=(1000,), dtype=tf.float64),
                                             lambda: np.random.normal(size=1000), ])
def test_from_tensors(weights_factory):
    weights = weights_factory()
    true_tensor = 42. * tf.ones(shape=(1000, 1), dtype=tf.float64)
    data = zfit.Data.from_tensor(obs='obs1', tensor=true_tensor,
                                      weights=weights)

    weights_data = data.weights
    x = data.value()
    x_np = zfit.run(x)
    if isinstance(weights, tf.Tensor):
        weights = zfit.run(weights)

    np.testing.assert_allclose(x_np, zfit.run(true_tensor))
    if weights is not None:
        weights_data = zfit.run(weights_data)
        np.testing.assert_allclose(weights_data, weights)
    else:
        assert weights is None


def test_overloaded_operators():
    data1 = create_data1()
    a = data1 * 5.
    np.testing.assert_array_equal(5 * example_data1, zfit.run(a))
    np.testing.assert_array_equal(example_data1, zfit.run(data1))
    data_squared = data1 * data1
    np.testing.assert_allclose(example_data1 ** 2, zfit.run(data_squared), rtol=1e-8)
    np.testing.assert_allclose(np.log(example_data1), zfit.run(tf.log(data1)), rtol=1e-8)


def test_sort_by_obs():
    data1 = create_data1()

    new_obs = (obs1[1], obs1[2], obs1[0])
    new_array = copy.deepcopy(example_data1)[:, np.array((1, 2, 0))]
    # new_array = np.array([new_array[:, 1], new_array[:, 2], new_array[:, 0]])
    assert data1.obs == obs1, "If this is not True, then the test will be flawed."
    with data1.sort_by_obs(new_obs):
        assert data1.obs == new_obs
        np.testing.assert_array_equal(new_array, zfit.run(data1.value()))
        new_array2 = copy.deepcopy(new_array)[:, np.array((1, 2, 0))]
        # new_array2 = np.array([new_array2[:, 1], new_array2[:, 2], new_array2[:, 0]])
        new_obs2 = (new_obs[1], new_obs[2], new_obs[0])
        with data1.sort_by_obs(new_obs2):
            assert data1.obs == new_obs2
            np.testing.assert_array_equal(new_array2, zfit.run(data1.value()))

        assert data1.obs == new_obs

    assert data1.obs == obs1
    np.testing.assert_array_equal(example_data1, zfit.run(data1.value()))


def test_subdata():
    data1 = create_data1()
    new_obs = (obs1[0], obs1[1])
    new_array = copy.deepcopy(example_data1)[:, np.array((0, 1))]
    # new_array = np.array([new_array[:, 0], new_array])
    with data1.sort_by_obs(obs=new_obs):
        assert data1.obs == new_obs
        np.testing.assert_array_equal(new_array, zfit.run(data1))
        new_array2 = copy.deepcopy(new_array)[:, 1]
        # new_array2 = np.array([new_array2[:, 1]])
        new_obs2 = (new_obs[1],)
        with data1.sort_by_obs(new_obs2):
            assert data1.obs == new_obs2
            np.testing.assert_array_equal(new_array2, zfit.run(data1.value())[:, 0])

            with pytest.raises(ValueError):
                with data1.sort_by_obs(obs=new_obs):
                    print(zfit.run(data1.value()))

    assert data1.obs == obs1
    np.testing.assert_array_equal(example_data1, zfit.run(data1.value()))


def test_data_range():
    data1 = np.array([[1., 2],
                      [0, 1],
                      [-2, 1],
                      [-1, -1],
                      [-5, 10]])
    # data1 = data1.transpose()
    obs = ['obs1', 'obs2']
    lower = ((0.5, 1), (-3, -2))
    upper = ((1.5, 2.5), (-1.5, 1.5))
    data_range = zfit.Space(obs=obs, limits=(lower, upper))
    cut_data1 = data1[np.array((0, 2)), :]

    dataset = zfit.Data.from_tensor(obs=obs, tensor=data1)
    value_uncut = dataset.value()
    np.testing.assert_equal(data1, zfit.run(value_uncut))
    with dataset.set_data_range(data_range):
        value_cut = dataset.value()
        np.testing.assert_equal(cut_data1, zfit.run(value_cut))
        np.testing.assert_equal(data1, zfit.run(value_uncut))  # check  that the original did NOT change

    np.testing.assert_equal(cut_data1, zfit.run(value_cut))
    np.testing.assert_equal(data1, zfit.run(dataset.value()))


def test_multidim_data_range():
    data1 = np.linspace((0, 5), (10, 15), num=11)
    data_true = np.linspace(10, 15, num=6)
    lower = ((5, 5),)
    upper = ((10, 15),)
    obs1 = "x"
    obs2 = "y"
    data_range = zfit.Space([obs1, obs2], limits=(lower, upper))
    dataset = zfit.Data.from_numpy(array=data1, obs=data_range)
    assert zfit.run(dataset.nevents) == 6
    with dataset.sort_by_obs(obs=obs1):
        assert zfit.run(dataset.nevents) == 6
    assert zfit.run(dataset.nevents) == 6
    with dataset.sort_by_obs(obs=obs2):
        assert zfit.run(dataset.nevents) == 6
        np.testing.assert_allclose(data_true, zfit.run(dataset.unstack_x()))

    data2 = np.linspace((0, 5), (10, 15), num=11)[:, 1]
    data_range = zfit.Space([obs1], limits=(5, 15))
    dataset = zfit.Data.from_numpy(array=data2, obs=data_range)
    assert zfit.run(dataset.nevents) == 11

import copy

import pytest

import numpy as np
import tensorflow as tf

import zfit

# for i in range(1):
#     print(i)
#     try:
#         func = tf.log(x) * tf.sin(x) * tf.reduce_mean(x ** 2 - tf.cos(x) ** 2) / tf.reduce_sum(x)
#         start = time.time()
#         result_grad = sess.run(tf.gradients(func, x))
#         result = sess.run(func)
#         end = time.time()
#         print("time needed", (end - start))
#     except tf.errors.OutOfRangeError:
#         print("finished at i = ", i)
#         break
#     print(np.shape(result))
#     print(result[:, :10])
#     print(result_grad)

obs1 = ('obs1', 'obs2', 'obs3')

example_data1 = np.random.random(size=(len(obs1), 7))
data1 = zfit.data.Data.from_numpy(obs=obs1, array=example_data1)


def test_from_root_iter():
    try:
        from skhep_testdata import data_path
    except ImportError:
        return  # TODO: install skhep_testdata for tests

    path_root = data_path("uproot-Zmumu.root")

    branches = [b'pt1', b'pt2']  # b needed currently -> uproot

    data = zfit.data.Data.from_root(path=path_root, treepath='events', branches=branches)

    data.initialize(sess=zfit.sess)
    x = data.value()


def test_from_root():
    try:
        from skhep_testdata import data_path
    except ImportError:
        return  # TODO: install skhep_testdata for tests

    path_root = data_path("uproot-Zmumu.root")

    branches = [b'pt1', b'pt2']  # b needed currently -> uproot

    data = zfit.data.Data.from_root(path=path_root, treepath='events', branches=branches)
    data.initialize(sess=zfit.sess)
    x = data.value()


def test_from_numpy():
    example_data = np.random.random(size=(len(obs1), 1000))
    data = zfit.data.Data.from_numpy(obs=obs1, array=example_data)
    # data.initialize(sess=zfit.sess)
    x = data.value()
    x_np = zfit.sess.run(x)
    np.testing.assert_array_equal(example_data, x_np)


def test_from_tensors():
    pass
    # data.initialize(sess=zfit.sess)
    # x = data.value()


def test_values():
    pass


def test_overloaded_operators():
    a = data1 * 5.
    np.testing.assert_array_equal(5 * example_data1, zfit.sess.run(a))
    np.testing.assert_array_equal(example_data1, zfit.sess.run(data1))
    data_squared = data1 * data1
    np.testing.assert_allclose(example_data1 ** 2, zfit.sess.run(data_squared), rtol=1e-8)
    np.testing.assert_allclose(np.log(example_data1), zfit.sess.run(tf.log(data1)), rtol=1e-8)


def test_sort_by_obs():
    new_obs = (obs1[1], obs1[2], obs1[0])
    new_array = copy.deepcopy(example_data1)
    new_array = np.array([new_array[1, :], new_array[2, :], new_array[0, :]])
    assert data1.obs == obs1, "If this is not True, then the test will be flawed."
    with data1.sort_by_obs(new_obs):
        assert data1.obs == new_obs
        np.testing.assert_array_equal(new_array, zfit.sess.run(data1.value()))
        new_array2 = copy.deepcopy(new_array)
        new_array2 = np.array([new_array2[1, :], new_array2[2, :], new_array2[0, :]])
        new_obs2 = (new_obs[1], new_obs[2], new_obs[0])
        with data1.sort_by_obs(new_obs2):
            assert data1.obs == new_obs2
            np.testing.assert_array_equal(new_array2, zfit.sess.run(data1.value()))

        assert data1.obs == new_obs

    assert data1.obs == obs1
    np.testing.assert_array_equal(example_data1, zfit.sess.run(data1.value()))


def test_subdata():
    new_obs = (obs1[0], obs1[1])
    new_array = copy.deepcopy(example_data1)
    new_array = np.array([new_array[0, :], new_array[1, :]])
    with data1.sort_by_obs(obs=new_obs):
        assert data1.obs == new_obs
        np.testing.assert_array_equal(new_array, zfit.sess.run(data1))
        new_array2 = copy.deepcopy(new_array)
        new_array2 = np.array([new_array2[1, :]])
        new_obs2 = (new_obs[1],)
        with data1.sort_by_obs(new_obs2):
            assert data1.obs == new_obs2
            np.testing.assert_array_equal(new_array2, zfit.sess.run(data1.value()))

            with pytest.raises(ValueError):
                with data1.sort_by_obs(obs=new_obs):
                    print(zfit.sess.run(data1.value()))

    assert data1.obs == obs1
    np.testing.assert_array_equal(example_data1, zfit.sess.run(data1.value()))

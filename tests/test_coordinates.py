#  Copyright (c) 2020 zfit
import numpy as np
import pytest
import tensorflow as tf

from zfit import z
from zfit.core.space_new import Coordinates
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


@pytest.mark.parametrize('graph', [True, False])
def test_reorder_x(graph):
    x = tf.constant([2, 3, 4, 1], dtype=tf.float64)

    def test():
        x_all = []
        obs = ['a', 'b', 'c', 'd']
        x_obs = ['b', 'c', 'd', 'a']
        func_obs = ['d', 'a', 'b', 'c', ]
        coords = Coordinates(obs)
        x_all.append(coords.reorder_x(x, x_obs=x_obs))
        x_all.append(coords.reorder_x(x, func_obs=func_obs))

        axes = [0, 1, 2, 3]
        x_axes = [1, 2, 3, 0]
        func_axes = [3, 0, 1, 2]
        coords = Coordinates(axes=axes)
        x_all.append(coords.reorder_x(x, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_axes=func_axes))

        coords = Coordinates(obs, axes)
        x_all.append(coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_obs=func_obs, func_axes=func_axes))

        coords = Coordinates(obs, axes)
        x_all.append(coords.reorder_x(x, x_obs=x_obs))
        x_all.append(coords.reorder_x(x, func_obs=func_obs))

        coords = Coordinates(obs)
        x_all.append(coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_obs=func_obs, func_axes=func_axes))

        with pytest.raises(ValueError):
            coords.reorder_x(x, x_axes=x_axes)
        with pytest.raises(ValueError):
            coords.reorder_x(x, func_axes=func_axes)

        coords = Coordinates(axes=axes)
        with pytest.raises(ValueError):
            coords.reorder_x(x, x_obs=x_obs)
        with pytest.raises(ValueError):
            coords.reorder_x(x, func_obs=func_obs)

        return x_all

    if graph:
        test = z.function(test)
    all_x = test()
    true_x = tf.constant([1, 2, 3, 4], dtype=tf.float64)
    for x in all_x:
        assert np.allclose(x, true_x)


@pytest.mark.parametrize('graph', [True, False])
def test_with_obs_or_axes(graph):
    x = tf.constant([2, 3, 4, 1], dtype=tf.float64)

    def test():
        x_all = []
        obs = ['a', 'b', 'c', 'd']
        obs1 = ['b', 'c', 'd', 'a']
        obs2 = ['d', 'a', 'b', 'c', ]
        coords_obs = Coordinates(obs)

        axes = [0, 1, 2, 3]
        axes1 = [1, 2, 3, 0]
        axes2 = [3, 0, 1, 2]
        coords_axes = Coordinates(axes=axes)

        coords = Coordinates(obs=obs, axes=axes)

        coords1 = coords.with_obs(obs1)
        coords_obs1 = coords_obs.with_obs(obs1)
        assert coords1.obs == obs1
        assert coords1.axes == axes1
        assert coords_obs1.obs == obs1

        coords2 = coords.with_obs(obs2)
        coords_obs2 = coords_obs.with_obs(obs2)
        assert coords2.obs == obs2
        assert coords2.axes == axes2
        assert coords_obs2.obs == obs2

        coords1 = coords.with_axes(axes1)
        coords_axes1 = coords_axes.with_axes(axes1)
        assert coords1.obs == obs1
        assert coords1.axes == axes1
        assert coords_axes1.axes == axes1

        coords2 = coords.with_axes(axes2)
        coords_axes2 = coords_axes.with_axes(axes2)
        assert coords2.obs == obs2
        assert coords2.axes == axes2
        assert coords_axes2.axes == axes2

    if graph:
        test = z.function(test)

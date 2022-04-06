#  Copyright (c) 2022 zfit
import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import z
from zfit.core.coordinates import Coordinates
from zfit.util.exception import AxesIncompatibleError, ObsIncompatibleError

coordinate_classes = [Coordinates, zfit.Space]


@pytest.mark.parametrize("graph", [True, False])
@pytest.mark.parametrize("testclass", coordinate_classes)
def test_reorder_x(graph, testclass):
    x = tf.constant([2, 3, 4, 1], dtype=tf.float64)

    def test():
        x_all = []
        obs = ["a", "b", "c", "d"]
        x_obs = ["b", "c", "d", "a"]
        func_obs = [
            "d",
            "a",
            "b",
            "c",
        ]
        coords = testclass(obs)
        x_all.append(coords.reorder_x(x, x_obs=x_obs))
        x_all.append(coords.reorder_x(x, func_obs=func_obs))

        axes = [0, 1, 2, 3]
        x_axes = [1, 2, 3, 0]
        func_axes = [3, 0, 1, 2]
        coords = testclass(axes=axes)
        x_all.append(coords.reorder_x(x, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_axes=func_axes))

        coords = testclass(obs=obs, axes=axes)
        x_all.append(coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_obs=func_obs, func_axes=func_axes))

        coords = testclass(obs=obs, axes=axes)
        x_all.append(coords.reorder_x(x, x_obs=x_obs))
        x_all.append(coords.reorder_x(x, func_obs=func_obs))

        coords = testclass(obs)
        x_all.append(coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes))
        x_all.append(coords.reorder_x(x, func_obs=func_obs, func_axes=func_axes))

        with pytest.raises(ValueError):
            coords.reorder_x(x, x_axes=x_axes)
        with pytest.raises(ValueError):
            coords.reorder_x(x, func_axes=func_axes)

        coords = testclass(axes=axes)
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


@pytest.mark.parametrize("graph", [True, False])
@pytest.mark.parametrize("testclass", coordinate_classes)
def test_with_obs_or_axes(graph, testclass):
    x = tf.constant([2, 3, 4, 1], dtype=tf.float64)

    x_all = []
    obs = ("a", "b", "c", "d")
    obs1 = ("b", "c", "d", "a")
    obs2 = ("d", "a", "b", "c")
    sub_obs1 = ("c", "a")
    super_obs1 = ("a", "y", "b", "v", "c", "d")

    axes = (0, 1, 2, 3)
    axes1 = (1, 2, 3, 0)
    axes2 = (3, 0, 1, 2)
    super_axes1 = (0, 6, 1, 9, 2, 3)
    sub_axes1 = (2, 0)

    coords_obs = testclass(obs)
    coords_axes = testclass(axes=axes)
    coords = testclass(obs=obs, axes=axes)

    coords1 = coords.with_obs(obs1)
    coords_obs = coords_obs.with_obs(obs1)
    assert coords1.obs == obs1
    assert coords1.axes == axes1
    assert coords_obs.obs == obs1

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

    coords1 = coords.with_autofill_axes(overwrite=True)
    coords_obs = coords_obs.with_autofill_axes()
    assert coords1.obs == obs
    assert coords1.axes == axes
    assert coords_obs.obs == obs1
    assert coords_obs.axes == axes
    with pytest.raises(AxesIncompatibleError):
        coords.with_autofill_axes(overwrite=False)

    coords_obs = [f"{i}" if i % 2 else obs2[int(i / 2)] for i in range(len(obs2) * 2)]
    coords_obs2 = coords.with_obs(obs=coords_obs, allow_superset=True)
    assert coords_obs2.obs == obs2
    assert coords_obs2.axes == axes2

    coords_axes = [
        i + 10 if i % 2 else axes2[int(i / 2)] for i in range(len(axes2) * 2)
    ]
    coords_axes2 = coords.with_axes(axes=coords_axes, allow_superset=True)
    assert coords_axes2.obs == obs2
    assert coords_axes2.axes == axes2

    coords_obs = [
        f"{i}" if i % 2 else sub_obs1[int(i / 2)] for i in range(len(sub_obs1) * 2)
    ]
    coords_obs2 = coords.with_obs(
        obs=coords_obs, allow_superset=True, allow_subset=True
    )
    assert coords_obs2.obs == sub_obs1
    assert coords_obs2.axes == sub_axes1
    with pytest.raises(ObsIncompatibleError):
        coords_obs2 = coords.with_obs(
            obs=coords_obs, allow_superset=False, allow_subset=True
        )
    with pytest.raises(ObsIncompatibleError):
        coords_obs2 = coords.with_obs(
            obs=coords_obs, allow_superset=True, allow_subset=False
        )

    coords_axes = [
        i + 10 if i % 2 else sub_axes1[int(i / 2)] for i in range(len(sub_axes1) * 2)
    ]
    coords_axes2 = coords.with_axes(
        axes=coords_axes, allow_superset=True, allow_subset=True
    )
    assert coords_axes2.obs == sub_obs1
    assert coords_axes2.axes == sub_axes1
    with pytest.raises(AxesIncompatibleError):
        coords_axes2 = coords.with_axes(
            axes=coords_axes, allow_superset=False, allow_subset=True
        )
    with pytest.raises(AxesIncompatibleError):
        coords_axes2 = coords.with_axes(
            axes=coords_axes, allow_superset=True, allow_subset=False
        )

    coords_obs = sub_obs1
    coords_obs2 = coords.with_obs(obs=coords_obs, allow_subset=True)
    assert coords_obs2.obs == sub_obs1
    assert coords_obs2.axes == sub_axes1
    with pytest.raises(ObsIncompatibleError):
        coords_obs2 = coords.with_obs(obs=coords_obs, allow_subset=False)

    coords_axes = sub_axes1
    coords_axes2 = coords.with_axes(
        axes=coords_axes, allow_superset=True, allow_subset=True
    )
    assert coords_axes2.obs == sub_obs1
    assert coords_axes2.axes == sub_axes1
    with pytest.raises(AxesIncompatibleError):
        coords_obs2 = coords.with_axes(axes=coords_axes, allow_subset=False)

    coords_obs = super_obs1
    coords_obs2 = coords.with_obs(
        obs=coords_obs, allow_subset=True, allow_superset=True
    )
    assert coords_obs2.obs == obs
    assert coords_obs2.axes == axes
    with pytest.raises(ObsIncompatibleError):
        coords_obs2 = coords.with_obs(obs=coords_obs, allow_superset=False)

    coords_axes = super_axes1
    coords_axes2 = coords.with_axes(
        axes=coords_axes, allow_superset=True, allow_subset=True
    )
    assert coords_axes2.obs == obs
    assert coords_axes2.axes == axes
    with pytest.raises(AxesIncompatibleError):
        coords_obs2 = coords.with_axes(axes=coords_axes, allow_superset=False)

    # check non-overlaping obs
    with pytest.raises(ObsIncompatibleError):
        coords_obs2.with_obs(["er", "te", "qwer", "fd", "asd"])

    with pytest.raises(AxesIncompatibleError):
        coords_obs2.with_axes(list(range(10, 15)))

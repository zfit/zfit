#  Copyright (c) 2023 zfit

import copy
import random

import numpy as np
import pytest

import zfit
from zfit.core.space import Space, convert_to_space
from zfit.util.exception import (
    CoordinatesUnderdefinedError,
    LimitsIncompatibleError,
    LimitsUnderdefinedError,
    ShapeIncompatibleError,
)

lower11 = 1
lower12 = 4
upper11 = 3
upper12 = 7
limit11 = lower11, upper11
limit12 = lower12, upper12
limit11_true = copy.deepcopy(limit11)
limit12_true = copy.deepcopy(limit12)
limit11_area = 2
limit12_area = 3
limit11_axes = (1,)
limit12_axes = (1,)
limit11_obs = ("obs1",)
limit12_obs = ("obs1",)
limit11_axes_true = limit11_axes
limit12_axes_true = limit12_axes
space11 = Space(limits=limit11, axes=limit11_axes)
space12 = Space(limits=limit12, axes=limit12_axes)
space1 = space11 + space12
space11_obs = Space(obs=limit11_obs, limits=limit11)
space12_obs = Space(obs=limit12_obs, limits=limit12)
space1_obs = space11_obs + space12_obs
# arguments1 = (space1, lower1, upper1, limit1_true, limit1_axes, limit1_areas, 2)
arguments1 = []

lower2 = (1, 2, 3)
upper2 = (2, 4, 6)
sub_lower2 = (1, 3)
sub_upper2 = (2, 6)
limit2 = lower2, upper2
sub_limit2 = sub_lower2, sub_upper2
limit2_areas = (6, 18)
sub_limit2_areas = (3, 1.8)
limit2_axes = (1, 5, 6)
sub_limit2_axes = (1, 6)
limit2_obs = ("obs1", "obs2", "obs3")
sub_limit2_obs = ("obs1", "obs3")

space2 = Space(limits=limit2, axes=limit2_axes)
space2_obs = Space(obs=limit2_obs, limits=limit2)
sub_space2 = Space(limits=sub_limit2, axes=sub_limit2_axes)
space2_subbed_axes = space2.get_subspace(axes=sub_limit2_axes)

arguments2 = (space2, lower2, upper2, limit2, limit2_axes, limit2_areas, 2)
sub_arguments2 = (
    sub_space2,
    sub_lower2,
    sub_upper2,
    sub_limit2,
    sub_limit2_axes,
    sub_limit2_areas,
    2,
)


@pytest.mark.parametrize("space1, space2", [[space2_subbed_axes, sub_space2]])
def test_equality(space1, space2):
    """
    Args:
        space1:
        space2:
    """
    assert space1.axes == space2.axes
    assert space1.obs == space2.obs
    np.testing.assert_allclose(space1.rect_limits, space2.rect_limits)
    assert zfit.run(space1.rect_area()) == pytest.approx(
        zfit.run(space2.rect_area()), rel=1e-8
    )


# @pytest.mark.skip  # eq missing
def test_sub_space():
    sub_space2_true_axes = Space(axes=sub_limit2_axes, limits=sub_limit2)
    assert sub_space2_true_axes == space2_subbed_axes

    sub_space2_true = Space(obs=sub_limit2_obs, limits=sub_limit2)
    space2_subbed = space2_obs.get_subspace(obs=sub_limit2_obs)
    assert space2_subbed == sub_space2_true


@pytest.mark.skip
@pytest.mark.parametrize(
    "space,lower, upper, limit, axes, areas, n_limits",
    [
        # arguments1,
        # arguments2,
        # sub_arguments2,
    ],
)
def test_space(space, lower, upper, limit, axes, areas, n_limits):
    """
    Args:
        space:
        lower:
        upper:
        limit:
        axes:
        areas:
        n_limits:
    """
    assert space.rect_area() == pytest.approx(sum(areas), rel=1e-8)
    # assert space.iter_areas() == pytest.approx(areas, rel=1e-8)
    # assert sum(space.iter_areas(rel=True)) == pytest.approx(1, rel=1e-7)

    assert space.axes == axes

    assert space.limits == limit

    # iter1_limits1, iter2_limits1 = space.iter_limits(as_tuple=True)
    iter1_limits1_space, iter2_limits1_space = space
    # assert iter1_limits1 == (lower[0], upper[0])
    # assert iter2_limits1 == (lower[1], upper[1])
    # assert iter1_limits1 == (iter1_limits1_space.lower[0], iter1_limits1_space.upper[0])
    # assert iter2_limits1 == (iter2_limits1_space.lower[0], iter2_limits1_space.upper[0])
    # assert space.n_limits == n_limits
    # assert iter1_limits1_space.n_limits == 1
    # assert iter2_limits1_space.n_limits == 1


@pytest.mark.parametrize(
    "space,obs",
    [
        # (space1_obs, limit11_obs),
        (space2_obs, limit2_obs)
    ],
)
def test_setting_axes(space, obs):
    """
    Args:
        space:
        obs:
    """
    lower, upper = space.rect_limits
    axes = space.axes
    new_obs = list(copy.deepcopy(obs))
    while len(obs) > 1 and new_obs == list(obs):
        random.shuffle(new_obs)
    new_obs = tuple(new_obs)
    true_lower = np.array(
        tuple(tuple(low[obs.index(o)] for o in new_obs) for low in lower)
    )
    true_upper = np.array(
        tuple(tuple(up[obs.index(o)] for o in new_obs) for up in upper)
    )
    new_axes = tuple(range(len(new_obs)))
    coords = Space(obs=new_obs, axes=new_axes)
    # obs_axes = OrderedDict((o, ax) for o, ax in zip(new_obs, new_axes))

    if len(obs) > 1:
        # make sure it was shuffled
        assert new_obs != obs
        assert new_axes != axes
        assert np.any(true_lower != lower)
        assert np.any(true_upper != upper)

    new_space = space.with_coords(coords)
    # check new object
    assert new_axes == new_space.axes
    assert new_obs == new_space.obs
    assert np.all(true_lower == new_space.lower)
    assert np.all(true_upper == new_space.upper)

    # check that old object didn't change
    assert axes == space.axes
    assert obs == space.obs
    assert np.all(lower == space.lower)
    assert np.all(upper == space.upper)

    assert axes == space.axes
    assert obs == space.obs
    assert np.all(lower == space.lower)
    assert np.all(upper == space.upper)


def test_exception():
    invalid_obs = (1, 4)
    with pytest.raises(TypeError):
        Space(obs=invalid_obs)
    with pytest.raises(CoordinatesUnderdefinedError):
        Space(obs=None, limits=limit2)
    with pytest.raises(CoordinatesUnderdefinedError):
        Space(axes=None, limits=limit2)
    with pytest.raises(ShapeIncompatibleError):  # one obs only, but two dims
        Space(obs="obs1", limits=(((1, 2),), ((2, 3),)))
    with pytest.raises(ShapeIncompatibleError):  # two obs but only 1 dim
        Space(obs=["obs1", "obs2"], limits=(((1,),), ((2,),)))
    with pytest.raises(ShapeIncompatibleError):  # one axis but two dims
        Space(axes=(1,), limits=(((1, 2),), ((2, 3),)))
    with pytest.raises(ShapeIncompatibleError):  # two axes but only 1 dim
        Space(axes=(1, 2), limits=(((1,),), ((2,),)))
    # with pytest.raises(ValueError):  # two obs, two limits but each one only 1 dim
    #     Space(obs=['obs1', 'obs2'], limits=(((1,), (2,)), ((2,), (3,))))


def test_dimensions():
    lower1, lower2 = 1, 2
    upper1, upper2 = 2, 3
    space = Space(
        obs=["obs1", "obs2"], limits=(((lower1, lower2),), ((upper1, upper2),))
    )
    assert space.n_obs == 2
    assert space.n_limits == 1
    lower, upper = space.limits
    low1 = lower[0][0]
    low2 = lower[0][1]
    up1 = upper[0][0]
    up2 = upper[0][1]
    assert low1 == lower1
    assert low2 == lower2
    assert up1 == upper1
    assert up2 == upper2

    with pytest.raises(RuntimeError):
        space.limit1d

    with pytest.raises(LimitsIncompatibleError):
        _ = Space(obs="obs1", limits=(((1,), (2,)), ((2,), (3,))))

    space = Space(obs=["obs1", "obs2"], limits=(((1, 5),), ((2, 13),)))
    assert space.n_obs == 2
    with pytest.raises(RuntimeError):
        space.limit1d

    space = Space(obs="obs1", limits=(((1,),), ((2,),)))
    assert space.n_obs == 1
    assert space.n_limits == 1

    lower1 = 1
    upper1 = 2
    space = Space(axes=(1,), limits=(((lower1,),), ((upper1,),)))
    assert space.n_obs == 1
    assert space.n_limits == 1
    lower, upper = space.limit1d
    assert lower == lower1
    assert upper == upper1


def test_autoconvert():
    with pytest.raises(LimitsUnderdefinedError):
        convert_to_space(obs=["obs1", "obs2"], limits=(((1, 2),), ((2, 3),)))

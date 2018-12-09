from collections import OrderedDict
import copy
import random
from unittest import TestCase

import pytest
import numpy as np

from zfit.core.limits import NamedSpace
from zfit.util.exception import ConversionError

lower1 = (1,), (4,)
upper1 = (3,), (7,)
limit1 = lower1, upper1
limit1_true = copy.deepcopy(limit1)
limit1_areas = (2, 3)
limit1_axes = (1,)
limit1_obs = ("obs1",)
limit1_axes_true = limit1_axes
space1 = NamedSpace(lower=lower1, upper=upper1, axes=limit1_axes)
space1_obs = NamedSpace(lower=lower1, upper=upper1, obs=limit1_obs)
arguments1 = (space1, lower1, upper1, limit1_true, limit1_axes, limit1_areas, 2)

lower2 = (1, 2, 3), (-4, -5, 5)
sub_lower2 = (1, 3), (-4, 5)
upper2 = (2, 4, 6), (-1, 5, 5.6)
sub_upper2 = (2, 6), (-1, 5.6)
limit2 = lower2, upper2
sub_limit2 = sub_lower2, sub_upper2
limit2_areas = (6, 18)
sub_limit2_areas = (3, 1.8)
limit2_axes = (1, 5, 6)
sub_limit2_axes = (1, 6)
limit2_obs = ('obs1', 'obs2', 'obs3')
sub_limit2_obs = ('obs1', 'obs3')

space2 = NamedSpace(lower=lower2, upper=upper2, axes=limit2_axes)
space2_obs = NamedSpace(obs=limit2_obs, lower=lower2, upper=upper2)
sub_space2 = NamedSpace(lower=sub_lower2, upper=sub_upper2, axes=sub_limit2_axes)
space2_subbed = space2.get_subspace(axes=sub_limit2_axes)

arguments2 = (space2, lower2, upper2, limit2, limit2_axes, limit2_areas, 2)
sub_arguments2 = (sub_space2, sub_lower2, sub_upper2, sub_limit2, sub_limit2_axes, sub_limit2_areas, 2)


@pytest.mark.parametrize("space1, space2", [
    [space2_subbed, sub_space2]
    ])
def test_equality(space1, space2):
    assert space1.axes == space2.axes
    assert space1.obs == space2.obs
    assert space1.limits == space2.limits
    assert space1.iter_areas() == pytest.approx(space2.iter_areas(), rel=1e-8)


@pytest.mark.parametrize("space,lower, upper, limit, axes, areas, n_limits",
                         [
                             arguments1,
                             arguments2,
                             sub_arguments2,
                             ])
def test_space(space, lower, upper, limit, axes, areas, n_limits):
    assert space.area() == pytest.approx(sum(areas), rel=1e-8)
    assert space.iter_areas() == pytest.approx(areas, rel=1e-8)

    assert space.axes == axes

    assert space.limits == limit

    iter1_limits1, iter2_limits1 = space.iter_limits(as_tuple=True)
    iter1_limits1_space, iter2_limits1_space = space.iter_limits(as_tuple=False)
    assert iter1_limits1 == (lower[0], upper[0])
    assert iter2_limits1 == (lower[1], upper[1])
    assert iter1_limits1 == (iter1_limits1_space.lower[0], iter1_limits1_space.upper[0])
    assert iter2_limits1 == (iter2_limits1_space.lower[0], iter2_limits1_space.upper[0])
    assert space.n_limits == n_limits
    assert iter1_limits1_space.n_limits == 1
    assert iter2_limits1_space.n_limits == 1


@pytest.mark.parametrize("space,obs",
                         [
                             (space1_obs, limit1_obs),
                             (space2_obs, limit2_obs)

                             ])
def test_setting_axes(space, obs):
    lower, upper = space.limits
    axes = space.axes
    new_obs = list(copy.deepcopy(obs))
    while len(obs) > 1 and new_obs == list(obs):
        random.shuffle(new_obs)
    new_obs = tuple(new_obs)
    true_lower = tuple(tuple(low[obs.index(o)] for o in new_obs) for low in lower)
    true_upper = tuple(tuple(up[obs.index(o)] for o in new_obs) for up in upper)
    new_axes = tuple(range(len(new_obs)))
    obs_axes = OrderedDict((o, ax) for o, ax in zip(new_obs, new_axes))

    if len(obs) > 1:
        # make sure it was shuffled
        assert new_obs != obs
        assert new_axes != axes
        assert true_lower != lower
        assert true_upper != upper

    new_space = space.with_obs_axes(obs_axes=obs_axes)
    # check new object
    assert new_axes == new_space.axes
    assert new_obs == new_space.obs
    assert true_lower == new_space.lower
    assert true_upper == new_space.upper

    # check that old object didn't change
    assert axes == space.axes
    assert obs == space.obs
    assert lower == space.lower
    assert upper == space.upper

    with space.set_obs_axes(obs_axes=obs_axes):
        assert new_axes == space.axes
        assert new_obs == space.obs
        assert true_lower == space.lower
        assert true_upper == space.upper

    assert axes == space.axes
    assert obs == space.obs
    assert lower == space.lower
    assert upper == space.upper

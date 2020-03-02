#  Copyright (c) 2020 zfit
import pytest
import numpy as np

import zfit
from zfit import z
from zfit.core.coordinates import Coordinates
from zfit.core.space_new import Space, Limit, ANY
from zfit.util.exception import CoordinatesUnderdefinedError, BehaviorUnderDiscussion
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester

obs1 = ('a', 'b', 'c', 'd', 'e')
obs2 = ('c', 'b', 'd', 'e', 'a')
axes1 = (0, 1, 2, 3, 4)
axes2 = (2, 1, 3, 4, 0)
coords1obs = Coordinates(obs1)
coords1 = Coordinates(obs1, axes1)
coords1mixed = Coordinates(obs1, axes2)
coords1axes = Coordinates(axes=axes1)
coords2obs = Coordinates(obs2)
coords2 = Coordinates(obs2, axes2)
coords2mixed = Coordinates(obs2, axes1)
coords2axes = Coordinates(axes=axes2)

limits1 = ([-2, -1, 0, 1, 2], [3, 4, 5, 6, 7])
limits2 = ([0, -1, 1, 2, -2], [5, 4, 6, 7, 3])

limits1tf = (z.convert_to_tensor([-2, -1, 0, 1, 2]), z.convert_to_tensor([3, 4, 5, 6, 7]))
limits2tf = (z.convert_to_tensor([0, -1, 1, 2, -2]), z.convert_to_tensor([5, 4, 6, 7, 3]))

limits1mixed_any = ([-2, ANY, 0, ANY, 2], [3, 4, 5, ANY, 7])
limits2mixed_any = ([0, ANY, ANY, 2, -2], [5, 4, ANY, 7, 3])

limits1any = ([ANY] * 5, [ANY] * 5)
limits2any = ([ANY] * 5, [ANY] * 5)

limits_to_test = [[limits1, limits2], [limits1tf, limits2tf],
                  [limits1mixed_any, limits2mixed_any], [limits1any, limits2any]]


def test_extract_limits():
    obs1 = ['a']
    space1 = Space('a', (0, 1))
    obs2 = ['b', 'c']
    limit2 = Limit(limit_fn=lambda x: x, rect_limits=([1, 2], [2, 3]), n_obs=2)
    obs3 = ['d', 'e', 'f']
    limits3_dict = {'obs': {ob: Limit((i, i + 10)) for i, ob in enumerate(obs3)}}
    space3 = Space(obs3, limits=limits3_dict)
    limits_dict = {'obs': {
        tuple(obs1): space1,
        tuple(obs2): limit2,
        tuple(obs3): space3,
    }
    }
    space = Space(obs1 + obs2 + obs3, limits_dict)

    extracted_limits = space._extract_limits(obs1)
    assert list(extracted_limits.values())[0] is space1
    extracted_limits = space._extract_limits(obs2)
    assert list(extracted_limits.values())[0] is limit2
    extracted_limits = space._extract_limits(obs3)
    assert list(extracted_limits.values())[0] is space3

    extracted_limits = space._extract_limits(obs3[0])
    # assert list(extracted_limits.values())[0] == limits3_dict[obs3[0]]
    # obs9 = obs3[0:2] + obs2
    # extracted_limits = space._extract_limits(obs9)
    # assert extracted_limits is limits3_dict[]


def test_rect_limits():
    obs1 = ['a']
    axes1 = [0]
    space1 = Space('a', (0, 1))
    space1_nolim = Space('a')
    assert not space1_nolim.has_limits
    assert space1.has_limits
    assert space1.has_rect_limits
    space1_lim = space1_nolim.with_limits((0, 1))
    space1_ax = Space(axes=0, limits=(0, 1))
    lower, upper = space1.rect_limits
    assert lower == 0
    assert upper == 1

    lower, upper = space1_ax.rect_limits
    assert lower == 0
    assert upper == 1

    lower, upper = space1_lim.rect_limits
    assert lower == 0
    assert upper == 1


@pytest.mark.parametrize('limits', limits_to_test)
def test_with_coords(limits):
    limits, limits2 = limits
    space1obs = zfit.Space(obs1, limits=limits)
    space1 = zfit.Space(obs1, limits=limits, axes=axes1)
    space1axes = zfit.Space(limits=limits, axes=axes1)
    space1mixed = zfit.Space(obs1, limits=limits, axes=axes2)
    limits1 = space1obs.rect_limits_np

    space2obs = zfit.Space(obs2, limits=limits2)
    space2 = zfit.Space(obs2, limits=limits2, axes=axes2)
    space2mixed = zfit.Space(obs2, limits=limits2, axes=axes1)
    space2axes = zfit.Space(limits=limits2, axes=axes2)
    limits2 = space2obs.rect_limits_np

    # define which space to use in this tests
    space_used = space1obs

    space = space_used.with_coords(coords1obs)
    assert space.obs == obs1
    assert space.axes == None
    np.testing.assert_equal(space.rect_limits_np, limits1)

    space = space_used.with_coords(coords2obs)
    assert space.obs == obs2
    assert space.axes == None
    np.testing.assert_equal(space.rect_limits_np, limits2)

    coords = coords2
    space = space_used.with_coords(coords)
    assert space.obs == coords.obs
    assert space.axes == coords.axes
    np.testing.assert_equal(space.rect_limits_np, limits2)

    with pytest.raises(CoordinatesUnderdefinedError):
        space = space_used.with_coords(coords2axes)

    # define which space to use in this tests
    space_used = space1

    space = space_used.with_coords(coords1obs)
    assert space.obs == obs1
    assert space.axes == axes1
    np.testing.assert_equal(space.rect_limits_np, limits1)

    space = space_used.with_coords(coords2obs)
    assert space.obs == obs2
    assert space.axes == axes2
    np.testing.assert_equal(space.rect_limits_np, limits2)

    coords = coords2
    space = space_used.with_coords(coords)
    assert space.obs == obs2
    assert space.axes == axes2
    np.testing.assert_equal(space.rect_limits_np, limits2)

    with pytest.raises(BehaviorUnderDiscussion):
        space_used.with_coords(coords2mixed)

    with pytest.raises(BehaviorUnderDiscussion):
        space_used.with_coords(coords1mixed)

    space = space_used.with_coords(coords2axes)
    assert space.obs == obs2
    assert space.axes == axes2
    np.testing.assert_equal(space.rect_limits_np, limits2)

    # define which space to use in this tests
    space_used = space1axes

    space = space_used.with_coords(coords1axes)
    assert space.obs == None
    assert space.axes == axes1
    np.testing.assert_equal(space.rect_limits_np, limits1)

    with pytest.raises(CoordinatesUnderdefinedError):
        space = space_used.with_coords(coords2obs)

    coords = coords2
    space = space_used.with_coords(coords)
    assert space.obs == coords.obs
    assert space.axes == coords.axes
    np.testing.assert_equal(space.rect_limits_np, limits2)

    space = space_used.with_coords(coords2axes)
    assert space.obs == None
    assert space.axes == coords.axes
    np.testing.assert_equal(space.rect_limits_np, limits2)

#  Copyright (c) 2021 zfit
import pytest
import tensorflow as tf

import zfit
from zfit import z
from zfit.core.coordinates import Coordinates
from zfit.core.space import ANY, Limit, Space
from zfit.util.exception import (CoordinatesUnderdefinedError,
                                 LimitsIncompatibleError)


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_vectors():
    Limit._experimental_allow_vectors = True
    yield
    Limit._experimental_allow_vectors = False


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

limits1limit = Limit(limits1, n_obs=5)
limits2limit = Limit(limits2, n_obs=5)

limits1space = Space(obs=obs1, axes=axes1, limits=limits1)
limits2space = Space(obs=obs2, axes=axes2, limits=limits2)

limit_fn = lambda x: x
limits1func = Space(obs=obs1, axes=axes1, limits=limit_fn, rect_limits=limits1)
limits2func = limits1func.with_obs(obs2)

limits3 = ([11, 12, 13, 14, 15], [31, 41, 51, 61, 71])
limits4 = ([13, 12, 14, 15, 11], [51, 41, 61, 71, 31])

limits1vector = ([[-21, -11, -91, 11, 21], [-22, -12, -92, 12, 22], [-23, -13, -93, 13, 23]],
                 [[31, 41, 51, 61, 71], [32, 42, 52, 62, 72], [33, 43, 53, 63, 73]])
limits2vector = ([[-91, -11, 11, 21, -21], [-92, -12, 12, 22, -22], [-93, -13, 13, 23, -23]],
                 [[51, 41, 61, 71, 31], [52, 42, 62, 72, 32], [53, 43, 63, 73, 33]])

limits1tf = (z.convert_to_tensor([-2, -1, 0, 1, 2]), z.convert_to_tensor([3, 4, 5, 6, 7]))
limits2tf = (z.convert_to_tensor([0, -1, 1, 2, -2]), z.convert_to_tensor([5, 4, 6, 7, 3]))

limits1mixed_any = ([-2, ANY, 0, ANY, 2], [3, 4, 5, ANY, 7])
limits2mixed_any = ([0, ANY, ANY, 2, -2], [5, 4, ANY, 7, 3])

limits1any = ([ANY] * 5, [ANY] * 5)
limits2any = ([ANY] * 5, [ANY] * 5)

limits_to_test = [
    [limits1, limits2],
    [limits1limit, limits2limit],
    [limits1space, limits2space],
    [limits1tf, limits2tf],
    [limits1mixed_any, limits2mixed_any],
    [limits1any, limits2any],
    [limits1vector, limits2vector],
    [{'multi': [limits1, limits3]}, {'multi': [limits2, limits4]}],
    [{'multi': [limits1any, limits2any]}, {'multi': [limits1any, limits2any]}],
    [{'multi': [limits1, limits2any]}, {'multi': [limits1any, limits2]}],
]


def test_illegal_bounds():
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = zfit.Space(['obs1', 'obs2'], ([-1, 4], [2, 3]))


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

    extracted_limits = space.get_limits(obs1)['obs']
    assert list(extracted_limits.values())[0] == space1
    extracted_limits = space.get_limits(obs2)['obs']
    assert list(extracted_limits.values())[0] == limit2
    extracted_limits = space.get_limits(obs3)['obs']
    assert list(extracted_limits.values())[0] == space3

    extracted_limits = space.get_limits(obs3[0])
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


def space_factory(*args, limits=None, **kwargs):
    """
    Args:
        *args:
        limits:
        **kwargs:
    """
    if isinstance(limits, dict):
        limit1, limit3 = limits['multi']
        space1 = zfit.Space(*args, limits=limit1, **kwargs)
        space3 = zfit.Space(*args, limits=limit3, **kwargs)
        return space1 + space3
    else:
        space = zfit.Space(*args, limits=limits, **kwargs)
        return space


@pytest.mark.parametrize('limits', limits_to_test)
def test_with_coords(limits):
    """
    Args:
        limits:
    """
    limits1, limits2 = limits

    space1obs = space_factory(obs1, limits=limits1)
    space1 = space_factory(obs1, limits=limits1, axes=axes1)
    space1axes = space_factory(limits=limits1, axes=axes1)
    # space1mixed = space_factory(obs1, limits=limits1, axes=axes2)

    space2obs = space_factory(obs2, limits=limits2)
    space2 = space_factory(obs2, limits=limits2, axes=axes2)
    space2axes = space_factory(limits=limits2, axes=axes2)

    # define which space to use in this tests
    space_used = space1obs

    space = space_used.with_coords(coords1obs)
    assert space == space1obs

    space = space_used.with_coords(coords2obs)
    assert space == space2obs

    coords = coords2
    space = space_used.with_coords(coords)
    assert space == space2

    with pytest.raises(CoordinatesUnderdefinedError):
        space = space_used.with_coords(coords2axes)

    space = space_used.with_coords(coords2mixed)
    assert space.obs == coords2mixed.obs
    assert space.axes == coords2mixed.axes

    # define which space to use in this tests
    space_used = space1

    space = space_used.with_coords(coords1obs)
    assert space == space1

    space = space_used.with_coords(coords2obs)
    assert space == space2

    space = space_used.with_coords(coords2)
    assert space == space2

    space = space_used.with_coords(coords2mixed)
    assert space.axes == coords2mixed.axes
    assert space.obs == coords2mixed.obs

    space = space_used.with_coords(coords1mixed)
    assert space.obs == coords1mixed.obs
    assert space.axes == coords1mixed.axes

    space = space_used.with_coords(coords2axes)
    assert space == space2

    # define which space to use in this tests
    space_used = space1axes

    space = space_used.with_coords(coords1axes)
    assert space == space1axes

    with pytest.raises(CoordinatesUnderdefinedError):
        space = space_used.with_coords(coords2obs)

    space = space_used.with_coords(coords2)
    assert space == space2

    space = space_used.with_coords(coords2axes)
    assert space == space2axes


@pytest.mark.parametrize('limits', limits_to_test)
def test_with_obs(limits):
    """
    Args:
        limits:
    """
    limits1, limits2 = limits

    space1obs = space_factory(obs1, limits=limits1)
    space1 = space_factory(obs1, limits=limits1, axes=axes1)
    space1axes = space_factory(limits=limits1, axes=axes1)
    using_space_as_lim = isinstance(limits1, Space)
    if isinstance(limits1, Space):
        limits1 = limits1.rect_limits
    space1mixed = space_factory(obs1, limits=limits1, axes=axes2)

    space2obs = space_factory(obs2, limits=limits2)
    space2 = space_factory(obs2, limits=limits2, axes=axes2)
    if isinstance(limits2, Space):
        limits2 = limits2.rect_limits
    space2axes = space_factory(limits=limits2, axes=axes2)
    space2mixed = space_factory(obs2, limits=limits1, axes=axes1)

    # define which space to use in this tests
    space_used = space1obs

    space = space_used.with_obs(obs1)
    assert space == space1obs

    space = space_used.with_obs(obs2)
    assert space == space2obs

    # define which space to use in this tests
    space_used = space1

    space = space_used.with_obs(obs1)
    assert space == space1

    space = space_used.with_obs(obs2)
    assert space == space2

    # define which space to use in this tests
    space_used = space1axes

    space = space_used.with_obs(obs1)
    assert space == space1

    space = space_used.with_obs(obs2)
    if using_space_as_lim:

        assert space == space2mixed
    else:
        assert space == space2mixed


@pytest.mark.parametrize('limits', limits_to_test)
def test_with_axes(limits):
    """
    Args:
        limits:
    """
    limits1, limits2 = limits

    space1obs = space_factory(obs1, limits=limits1)
    space1 = space_factory(obs1, limits=limits1, axes=axes1)
    space1axes = space_factory(limits=limits1, axes=axes1)
    using_space_as_lim = isinstance(limits1, Space)
    if isinstance(limits1, Space):
        limits1 = limits1.rect_limits
    space1mixed = space_factory(obs1, limits=limits1, axes=axes2)

    space2 = space_factory(obs2, limits=limits2, axes=axes2)
    if isinstance(limits2, Space):
        limits2 = limits2.rect_limits
    space2axes = space_factory(limits=limits2, axes=axes2)

    # define which space to use in this tests
    space_used = space1obs

    space = space_used.with_axes(axes1)
    assert space == space1

    space = space_used.with_axes(axes2)
    if using_space_as_lim:
        assert space == space1mixed
    else:
        assert space == space1mixed

    # define which space to use in this tests
    space_used = space1

    space = space_used.with_axes(axes1)
    assert space == space1

    space = space_used.with_axes(axes2)
    assert space == space2

    # define which space to use in this tests
    space_used = space1axes

    space = space_used.with_axes(axes1)
    assert space == space1axes

    space = space_used.with_axes(axes2)
    assert space == space2axes


@pytest.mark.parametrize('limits', limits_to_test)
def test_space_add(limits):
    """
    Args:
        limits:
    """
    limits1, limits2 = limits

    space1obs = space_factory(obs1, limits=limits1)
    space1 = space_factory(obs1, limits=limits1, axes=axes1)
    space1axes = space_factory(limits=limits1, axes=axes1)
    if isinstance(limits1, Space):
        limits1 = limits1.rect_limits
    space1mixed = space_factory(obs1, limits=limits1, axes=axes2)

    space2obs = space_factory(obs2, limits=limits2)
    space2 = space_factory(obs2, limits=limits2, axes=axes2)

    space2axes = space_factory(limits=limits2, axes=axes2)
    if isinstance(limits2, Space):
        limits2 = limits2.rect_limits
    space2mixed = space_factory(obs2, limits=limits2, axes=axes1)

    space = space1 + space2
    assert space.obs == obs1
    assert space.axes == axes1
    for spm, sp in zip(space, [*space1, *space2]):
        assert spm == sp

    space = space2 + space1
    assert space.obs == obs2
    assert space.axes == axes2
    for spm, sp in zip(space, [*space2, *space1]):
        assert spm == sp

    space = space1obs + space2
    assert space.obs == obs1
    assert space.axes == None

    space = space1axes + space2
    assert space.obs == None
    assert space.axes == axes1

    space = space2obs + space1
    assert space.obs == obs2
    assert space.axes == None

    space = space2axes + space1
    assert space.obs == None
    assert space.axes == axes2


def test_combine_spaces():
    shift = 30

    lower1, upper1 = [0, 1], [14, 13]
    lower1b, upper1b = [0 + shift, 1 + shift], [14 + shift, 13 + shift]
    lower2, upper2 = [-4, 1], [10, 13]
    lower2b, upper2b = [-4 + shift, 1 + shift], [10 + shift, 13 + shift]
    lower3, upper3 = [9, 1, 0], [11, 13, 14]
    obs1 = ['x', 'y']
    space1a = zfit.Space(obs1, limits=(lower1, upper1))
    space1b = zfit.Space(obs1, limits=(lower1b, upper1b))
    obs2 = ['z', 'y']
    space2a = zfit.Space(obs2, limits=(lower2, upper2))
    space2b = zfit.Space(obs2, limits=(lower2b, upper2b))
    obs3 = ['a', 'y', 'x']
    space3 = zfit.Space(obs3, limits=(lower3, upper3))
    space3inc = zfit.Space(obs3, limits=(lower3, upper3[::-1]))

    lower12 = [lower1[0], lower1[1], lower2[0]]
    upper12 = [upper1[0], upper1[1], upper2[0]]
    space12a = zfit.Space(('x', 'y', 'z'), limits=(lower12, upper12))
    space12b = zfit.Space(('x', 'y', 'z'), limits=([low + shift for low in lower12], [up + shift for up in upper12]))

    # space3 = zfit.Space(('x', 'y', 'z'), limits=([lower1[0], lower1[1], lower2[0]], [upper1[0], upper1[1], upper2[0]]))
    obs2inv = space2a.with_obs(['y', 'z'])

    space = space1a * space2a
    assert space == space12a
    assert space == space12a * obs2inv
    assert space == space12a * obs2inv * space12a

    space = space3 * space1a
    assert space == space3

    with pytest.raises(LimitsIncompatibleError):
        space1a * space3inc

    space12 = space12a + space12b
    space1 = space1a + space1b
    space2 = space2a + space2b
    space = space1 * space2
    assert space == space12

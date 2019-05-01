#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester


import os

import pytest

from zfit.util.exception import (LimitsIncompatibleError, LimitsNotSpecifiedError, ShapeIncompatibleError,
                                 SpaceIncompatibleError, )

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import zfit
from zfit.core.dimension import add_spaces, limits_consistent, limits_overlap, combine_spaces
from zfit.core.testing import setup_function, teardown_function, tester

obs = ['obs' + str(i) for i in range(4)]
space1 = zfit.Space(obs=obs)
space2 = zfit.Space(obs=reversed(obs))
space3 = zfit.Space(obs=obs)
space4 = zfit.Space(obs=obs[0:2])

lower1d_1 = ((1,),)
upper1d_1 = ((2,),)

lower1d_12 = ((3,),)
upper1d_12 = ((4,),)

lower1d_22 = ((5,),)
upper1d_22 = ((6,),)

combined_lim_1d_12_and_22 = (((3,), (5,)), ((4,), (6,)))

lower1d_2 = ((1,), (3,))
upper1d_2 = ((2,), (4,))

# lower1d_2 = ((1,), (3,))
# upper1d_2 = ((2,), (4,))

lower2d_1 = ((1.5, 3),)
upper2d_1 = ((2, 4),)

lower2d_2 = ((1, 3),)
upper2d_2 = ((2, 4),)

space1d_1 = zfit.Space(obs=obs[0], limits=(lower1d_1, upper1d_1))
space1d_12 = zfit.Space(obs=obs[1], limits=(lower1d_12, upper1d_12))
space1d_22 = zfit.Space(obs=obs[1], limits=(lower1d_22, upper1d_22))
space1d_2 = zfit.Space(obs=obs[0], limits=(lower1d_2, upper1d_2))

space2d_1 = zfit.Space(obs=obs[:2], limits=(lower2d_1, upper2d_1))
space2d_2 = zfit.Space(obs=obs[:2], limits=(lower2d_2, upper2d_2))


def test_check_n_obs():
    with pytest.raises(SpaceIncompatibleError):
        zfit.pdf.Gauss(1., 4., obs=space2d_1)


def test_combine_spaces():
    combined_space = combine_spaces(spaces=[space1d_1, space1d_12])
    combined_space2 = space1d_1 * space1d_12
    assert combined_space == space2d_2
    assert combined_space2 == space2d_2
    assert combine_spaces(spaces=[space2d_2, space2d_2, space2d_2]) == space2d_2
    none_limits_space = combine_spaces(spaces=[space1, space1, space1])
    assert none_limits_space == space1  # with None limits
    assert none_limits_space.limits is None
    with pytest.raises(LimitsNotSpecifiedError):
        combine_spaces(spaces=[space2d_2, space1])

    # same but different syntax
    combined_space = space1d_1.combine(space1d_12)
    assert combined_space == space2d_2
    assert space2d_2.combine([space2d_2, space2d_2]) == space2d_2
    none_limits_space = space1.combine([space1, space1])
    assert none_limits_space == space1  # with None limits
    assert none_limits_space.limits is None
    with pytest.raises(LimitsNotSpecifiedError):
        space2d_2.combine(space1)


def test_add_spaces():
    with pytest.raises(ValueError):
        assert add_spaces(spaces=space1)
    with pytest.raises(ValueError):
        assert add_spaces(spaces=[space1])
    with pytest.raises(ValueError):
        assert add_spaces(spaces=[])
    with pytest.raises(LimitsIncompatibleError):
        assert add_spaces(spaces=[space2d_1, space2d_2])

    assert add_spaces(spaces=[space1, space2]) == space1
    assert add_spaces(spaces=[space1, space2, space3]) == space1
    assert not add_spaces(spaces=[space1, space2, space3, space4])

    assert space1 + space2 == space1
    assert space1 + space2 + space3 == space1
    assert not space1.add([space2, space3, space4])

    assert not add_spaces(spaces=(space1d_2, space2d_1))
    assert add_spaces(spaces=(space1d_2, space1d_1)).limits == space1d_2.limits

    assert add_spaces(spaces=[space1d_12, space1d_22]).limits == combined_lim_1d_12_and_22
    assert add_spaces(spaces=[space1d_12, space1d_12, space1d_22, space1d_12]).limits == combined_lim_1d_12_and_22
    assert add_spaces(spaces=[space1d_12, space1d_12, space1d_12]).limits == (lower1d_12, upper1d_12)


def test_limits_consistent():
    assert limits_consistent(spaces=[space1, space2])
    assert limits_consistent(spaces=[space1, space2, space3])
    assert limits_consistent(spaces=[space1, space2, space3, space4])

    assert limits_consistent(spaces=[space1d_1, space1d_1])
    assert limits_consistent(spaces=[space1d_1, space1d_12])
    assert not limits_consistent(spaces=[space1d_1, space1d_2])
    assert not limits_consistent(spaces=[space1d_1, space2d_1])
    assert limits_consistent(spaces=[space1d_1, space2d_2])
    assert limits_consistent(spaces=[space1d_1, space2d_2, space1d_12])
    assert not limits_consistent(spaces=[space1d_1, space2d_1, space1d_12])


def test_limits_overlap():
    assert not limits_overlap([space1d_1, space1d_2], allow_exact_match=True)
    assert limits_overlap([space1d_1, space1d_2])
    assert limits_overlap([space1d_1, space1d_2, space1d_12])
    assert limits_overlap([space1d_1, space2d_2, space1d_12])
    assert not limits_overlap([space1d_1, space2d_2, space1d_12], allow_exact_match=True)

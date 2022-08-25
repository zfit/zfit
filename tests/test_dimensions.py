#  Copyright (c) 2022 zfit

import pytest

import zfit
from zfit.core.dimension import limits_overlap, obs_subsets
from zfit.core.space import combine_spaces, limits_consistent
from zfit.util.exception import (
    LimitsNotSpecifiedError,
    MultipleLimitsNotImplemented,
    ObsIncompatibleError,
    SpaceIncompatibleError,
)

obs = ["obs" + str(i) for i in range(4)]
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

lower1d_2 = ((1,),)
upper1d_2 = ((2,),)

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
        zfit.pdf.Gauss(1.0, 4.0, obs=space2d_1)


def test_combine_spaces2():
    combined_space = combine_spaces(space1d_1, space1d_12)
    combined_space2 = space1d_1 * space1d_12
    assert combined_space == space2d_2
    assert combined_space2 == space2d_2
    assert combine_spaces(space2d_2, space2d_2, space2d_2) == space2d_2
    none_limits_space = combine_spaces(space1, space1, space1)
    assert none_limits_space == space1  # with None limits
    assert not none_limits_space.limits_are_set
    with pytest.raises(LimitsNotSpecifiedError):
        combine_spaces(space2d_2, space1)

    # same but different syntax
    combined_space = space1d_1.combine(space1d_12)
    assert combined_space == space2d_2
    assert space2d_2.combine(space2d_2, space2d_2) == space2d_2
    none_limits_space = space1.combine(space1, space1)
    assert none_limits_space == space1  # with None limits
    assert not none_limits_space.limits_are_set
    with pytest.raises(LimitsNotSpecifiedError):
        space2d_2.combine(space1)


def test_add_spaces2():
    from zfit.core.space import add_spaces as add_spaces

    assert add_spaces(space1, space2) == space1
    assert add_spaces(space1, space2, space3) == space1
    with pytest.raises(ObsIncompatibleError):
        add_spaces(space1, space2, space3, space4)

    assert space1 + space2 == space1
    assert space1 + space2 + space3 == space1
    with pytest.raises(ObsIncompatibleError):
        space1.add(space2, space3, space4)

    with pytest.raises(ObsIncompatibleError):
        add_spaces(space1d_2, space2d_1)
    with pytest.raises(MultipleLimitsNotImplemented):
        add_spaces(space1d_2, space1d_1).limits


def test_limits_consistent():
    assert limits_consistent(spaces=[space1, space2])
    assert limits_consistent(spaces=[space1, space2, space3])
    assert limits_consistent(spaces=[space1, space2, space3, space4])

    assert limits_consistent(spaces=[space1d_1, space1d_1])
    assert limits_consistent(spaces=[space1d_1, space1d_12])
    assert not limits_consistent(spaces=[space1d_1, space2d_1])
    assert limits_consistent(spaces=[space1d_1, space2d_2])
    assert limits_consistent(spaces=[space1d_1, space2d_2, space1d_12])
    assert not limits_consistent(spaces=[space1d_1, space2d_1, space1d_12])


def test_limits_overlap():
    assert not limits_overlap([space1d_1, space1d_2], allow_exact_match=True)
    assert limits_overlap([space1d_1, space1d_2])
    assert limits_overlap([space1d_1, space1d_2, space1d_12])
    assert limits_overlap([space1d_1, space2d_2, space1d_12])
    assert not limits_overlap(
        [space1d_1, space2d_2, space1d_12], allow_exact_match=True
    )


def test_obs_subsets():
    space1 = zfit.Space("obs1")
    space2 = zfit.Space(["obs1", "obs2", "obs4"])
    space5 = zfit.Space(["obs3"])
    space6 = zfit.Space("obs6")
    space4 = zfit.Space(["obs3", "obs7"])
    space3 = zfit.Space(["obs4", "obs5"])
    spaces = [space1, space2, space5, space6, space4, space3]
    obs_subset = obs_subsets(spaces)

    obs1comb = frozenset(["obs1", "obs2", "obs4", "obs5"])
    obs37comb = frozenset(["obs3", "obs7"])
    obs6 = frozenset(["obs6"])
    true_obs = {
        obs1comb,
        obs37comb,
        obs6,
    }

    assert true_obs == set(obs_subset.keys())
    assert obs_subset[obs1comb] == [space1, space2, space3]
    assert obs_subset[obs37comb] == [space5, space4]
    assert obs_subset[obs6] == [space6]

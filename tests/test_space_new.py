#  Copyright (c) 2020 zfit
from zfit.core.space_new import Space, Limit


def test_extract_limits():
    obs1 = ['a']
    space1 = Space('a', (0, 1))
    obs2 = ['b', 'c']
    limit2 = Limit(limit_fn=lambda x: x, rect_limits=([1, 2], [2, 3]), n_obs=2)
    obs3 = ['d', 'e', 'f']
    limits3_dict = {ob: Limit((i, i + 10)) for i, ob in enumerate(obs3)}
    space3 = Space(obs3, limits=limits3_dict)
    limits_dict = {
        tuple(obs1): space1,
        tuple(obs2): limit2,
        tuple(obs3): space3,
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

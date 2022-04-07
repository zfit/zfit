#  Copyright (c) 2022 zfit

import zfit

# Addition of limit with the same observables
simple_limit1 = zfit.Space(obs="obs1", limits=(-5, 1))
simple_limit2 = zfit.Space(obs="obs1", limits=(3, 7.5))

added_limits = simple_limit1 + simple_limit2
# OR equivalently
added_limits = simple_limit1.add(simple_limit2)

# multiplication of limits with different observables
first_limit_lower = (-5, 6)
first_limit_upper = (5, 10)

second_limit_lower = (7, 12)
second_limit_upper = (9, 15)

space1 = zfit.Space(obs=["obs1", "obs2"], limits=(first_limit_lower, first_limit_upper))
space2 = zfit.Space(
    obs=["obs3", "obs4"], limits=(second_limit_lower, second_limit_upper)
)

space4 = space1 * space2

assert space4.obs == ("obs1", "obs2", "obs3", "obs4")
assert space4.n_obs == 4

# retreiving the limits
# for a one dimensional space, we can use `limit1d` as a shortcut
lower, upper = simple_limit1.limit1d

# However, for a higher dimensional space, this won't work. We can use `space.limits`, which is equivalent to
# (space.lower, space.upper).
lower2d, upper2d = space1.limits

# these have shape (nevents, nobs), whereas nevents is usually just 1
assert lower2d.shape == (1, space1.n_obs)

# order to retrieve now the same limits as from the first space, we can use array notation [:, i]
lower1_from2d = lower2d[:, 0]
upper1_from2d = upper2d[:, 0]

assert lower1_from2d == first_limit_lower[0]
assert upper1_from2d == first_limit_upper[0]

#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np

import zfit

# create a multidimensional space
xobs = zfit.Space("xobs", -4, 4)
yobs = zfit.Space("yobs", -3, 5)
zobs = zfit.Space("z", -2, 4)
obs = xobs * yobs * zobs
assert obs.n_obs == 3

# retrieve a subspace
xobs = obs.with_obs("xobs")
yxobs = obs.with_obs(["yobs", "xobs"])  # note the change in order

# retrieve the limits
# they have been overcomplicated in the past, now they are simple. Just access the limits via space.v1
limits = obs.v1.limits  # (lower, upper)
lower = obs.v1.lower
upper = obs.v1.upper

assert np.all(lower == [-4, -3, -2])
assert np.all(upper == [4, 5, 4])

# the volume is the product of the volumes of the individual spaces
volume = obs.volume
assert volume == xobs.volume * yobs.volume * zobs.volume
assert xobs.volume == xobs.v1.upper - xobs.v1.lower

# for example, creating a linspace object is simple
x = np.linspace(xobs.v1.lower, xobs.v1.upper, 1000)
# or
x = np.linspace(*xobs.v1.limits, 1000)

# or even in 3D
x = np.linspace(*obs.v1.limits, 1000)
assert x.shape == (1000, 3)

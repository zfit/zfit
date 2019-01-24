from typing import Iterable
from contextlib import ExitStack
import functools

from ..util.container import convert_to_container
from .interfaces import ZfitDimensional
from ..util import ztyping


class BaseDimensional(ZfitDimensional):

    @property
    def obs(self) -> ztyping.ObsTypeReturn:
        return self.space.obs

    @property
    def axes(self) -> ztyping.AxesTypeReturn:
        return self.space.axes

    @property
    def n_obs(self) -> int:
        return self.space.n_obs


@functools.lru_cache(maxsize=500)
def get_same_obs(obs):
    deps = [set() for _ in range(len(obs))]
    for i, ob in enumerate(obs):
        for j, other_ob in enumerate(obs[i + 1:]):
            if not set(ob).isdisjoint(other_ob):
                deps[i].add(i)
                deps[i].add(j + i + 1)
                deps[j + i + 1].add(i)

    deps = tuple(tuple(dep) for dep in deps)

    return deps


def is_combinable(spaces):
    pass


def is_addable(spaces: Iterable["zfit.Space"], check_limits=True):
    spaces = convert_to_container(spaces)
    if len(spaces) <= 1:
        raise ValueError("Need at least two spaces to test addability.")
    obs = frozenset(frozenset(space.obs) for space in spaces)

    if len(obs) != 1:
        return False

    if check_limits:
        obs1 = spaces[0].obs
        spaces = [space.with_obs_axes(obs=obs) if not space.obs == obs1 else space for space in spaces]

        limits = frozenset(space.limits for space in spaces)
        if len(limits) != 1:
            return False

    return True

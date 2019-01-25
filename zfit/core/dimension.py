from typing import Iterable
from contextlib import ExitStack
import functools

import zfit
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


def add_spaces(spaces: Iterable["zfit.Space"]):
    spaces = convert_to_container(spaces)
    if len(spaces) <= 1:
        raise ValueError("Need at least two spaces to be added.")  # TODO: allow? usecase?
    obs = frozenset(frozenset(space.obs) for space in spaces)

    if len(obs) != 1:
        return False

    obs1 = spaces[0].obs
    spaces = [space.with_obs_axes(obs=obs) if not space.obs == obs1 else space for space in spaces]

    limits = frozenset(space.limits for space in spaces)
    if len(limits) != 1:
        return False

    return True  # TODO


def common_obs(spaces):
    spaces = convert_to_container(spaces, container=tuple)
    all_obs = []
    for space in spaces:
        for ob in space.obs:
            if ob not in all_obs:
                all_obs.append(ob)
    return all_obs


def limits_consistent(spaces: Iterable["zfit.Space"]):
    spaces = convert_to_container(spaces, container=tuple)
    if len(spaces) <= 1:
        raise ValueError("Need at least two spaces to test limit consistency.")  # TODO: allow? usecase?

    all_obs = common_obs(spaces=spaces)
    all_lower = []
    all_upper = []
    spaces = [space.with_obs(all_obs) for space in spaces]
    for space in spaces:
        if space.limits is None:
            continue
        lowers, uppers = space.limits
        lower = [tuple(low[space.obs.index(ob)] for low in lowers) if ob in space.obs else None for ob in all_obs]
        upper = [tuple(up[space.obs.index(ob)] for up in uppers) if ob in space.obs else None for ob in all_obs]
        all_lower.append(lower)
        all_upper.append(upper)

    all_limits_to_check = all_lower, all_upper
    for limits_to_check in all_limits_to_check:
        for index in range(len(all_obs)):
            current_lower = None
            for limit in limits_to_check:
                lim = limit[index]
                if lim is not None:
                    if current_lower is None:
                        current_lower = lim
                    elif current_lower != lim:
                        return False
    return True

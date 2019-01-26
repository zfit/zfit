from typing import Iterable, Union, List
from contextlib import ExitStack
import functools

import numpy as np

import zfit
from zfit.util.exception import SpaceIncompatibleError, DueToLazynessNotImplementedError
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
    if limits_overlap(spaces=spaces):
        raise SpaceIncompatibleError("")

    return True  # TODO


def limits_overlap(spaces: Union["zfit.Space", Iterable["zfit.Space"]], allow_exact_match: bool = False) -> bool:
    """Check if _any_ of the limits of `spaces` overlaps with _any_ other of `spaces`.

    This also checks multiple limits within one space. If `allow_exact_match` is set to true, then
    an *exact* overlap of limits is allowed.


    Args:
        spaces (Iterable[zfit.Space]):
        allow_exact_match (bool): An exact overlap of two limits is counted as "not overlapping".
            Example: limits from -1 to 3 and 4 to 5 to *NOT* overlap with the limits 4 to 5 *iff*
            `allow_exact_match` is True.

    Returns:
        bool: if there are overlapping limits.
    """
    # TODO(Mayou36): add approx comparison global in zfit
    eps = 1e-8  # epsilon for float comparisons
    spaces = convert_to_container(spaces, container=tuple)
    all_obs = common_obs(spaces=spaces)
    for obs in all_obs:
        lowers = []
        uppers = []
        for space in spaces:
            if not space.limits or obs not in space.obs:
                continue
            else:
                index = space.obs.index(obs)

            for lower, upper in space.iter_limits(as_tuple=True):
                low = lower[index]
                up = upper[index]

                for other_lower, other_upper in zip(lowers, uppers):
                    if allow_exact_match and np.allclose(other_lower, low) and np.allclose(other_upper, up):
                        continue
                    # TODO(Mayou36): tolerance? add global flags?
                    low_overlaps = other_lower - eps < low < other_upper + eps
                    up_overlaps = other_lower - eps < up < other_upper + eps
                    overlap = low_overlaps or up_overlaps
                    if overlap:
                        return True
                lowers.append(low)
                uppers.append(up)
    return False


def common_obs(spaces: ztyping.SpaceOrSpacesTypeInput) -> List[str]:
    """Extract the union of `obs` from `spaces` in the order of `spaces`.

    For example:
        | space1.obs: ['obs1', 'obs3']
        | space2.obs: ['obs2', 'obs3', 'obs1']
        | space3.obs: ['obs2']

        returns ['obs1', 'obs3', 'obs2']

    Args:
        spaces (): Spaces to extract the obs from

    Returns:
        List[str]: The observables as `str`
    """
    spaces = convert_to_container(spaces, container=tuple)
    all_obs = []
    for space in spaces:
        for ob in space.obs:
            if ob not in all_obs:
                all_obs.append(ob)
    return all_obs


def limits_consistent(spaces: ztyping.SpaceOrSpacesTypeInput):
    """Check if space limits are the *exact* same in each obs they are defined and therefore are compatible.

    In this case, if a space has several limits, e.g. from -1 to 1 and from 2 to 3 (all in the same observable),
    to be consistent with this limits, other limits have to have (in this obs) also the limits
    from -1 to 1 and from 2 to 3. Only having the limit -1 to 1 _or_ 2 to 3 is considered _not_ consistent.

    This function is useful to check if several spaces with *different* observables can be _combined_.

    Args:
        spaces (List[zfit.core.limits.Space]):

    Returns:
        bool:
    """
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
            current_limit = None
            for limit in limits_to_check:
                lim = limit[index]
                if lim is not None:
                    if current_limit is None:
                        current_limit = lim
                    elif not np.allclose(current_limit, lim):
                        return False

    return not limits_overlap(spaces=spaces, allow_exact_match=True)

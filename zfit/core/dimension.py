from typing import Iterable, Union, List
from contextlib import ExitStack
import functools

import numpy as np

import zfit
from zfit.util.exception import (SpaceIncompatibleError, DueToLazynessNotImplementedError, LimitsIncompatibleError,
                                 LimitsNotSpecifiedError, )
from ..util.container import convert_to_container
from .interfaces import ZfitDimensional
from ..util import ztyping


class BaseDimensional(ZfitDimensional):

    def _check_n_obs(self, space):
        if self._N_OBS is not None:
            if len(space.obs) != self._N_OBS:
                raise SpaceIncompatibleError("Exactly {} obs are allowed, {} are given.".format(self._N_OBS, space.obs))

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_N_OBS"):
            cls._N_OBS = None

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
    """Add two spaces and merge their limits if possible or return False.

    Args:
        spaces (Iterable[:py:class:`~zfit.Space`]):

    Returns:
        Union[None, :py:class:`~zfit.Space`, bool]:

    Raises:
        LimitsIncompatibleError: if limits of the `spaces` cannot be merged because they overlap
    """
    spaces = convert_to_container(spaces)
    if len(spaces) <= 1:
        raise ValueError("Need at least two spaces to be added.")  # TODO: allow? usecase?
    obs = frozenset(frozenset(space.obs) for space in spaces)

    if len(obs) != 1:
        return False

    obs1 = spaces[0].obs
    spaces = [space.with_obs(obs=obs) if not space.obs == obs1 else space for space in spaces]

    if limits_overlap(spaces=spaces, allow_exact_match=True):
        raise LimitsIncompatibleError("Limits of spaces overlap, cannot merge spaces.")

    lowers = []
    uppers = []
    for space in spaces:
        if space.limits is None:
            continue
        for lower, upper in space.iter_limits(as_tuple=True):
            for other_lower, other_upper in zip(lowers, uppers):
                lower_same = np.allclose(lower, other_lower)
                upper_same = np.allclose(upper, other_upper)
                assert not lower_same ^ upper_same, "Bug, please report as issue. limits_overlap did not catch right."
                if lower_same and upper_same:
                    break
            else:
                lowers.append(lower)
                uppers.append(upper)
    lowers = tuple(lowers)
    uppers = tuple(uppers)
    if len(lowers) == 0:
        limits = None
    else:
        limits = lowers, uppers
    new_space = zfit.Space(obs=spaces[0].obs, limits=limits)
    return new_space


def limits_overlap(spaces: ztyping.SpaceOrSpacesTypeInput, allow_exact_match: bool = False) -> bool:
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
                    low_overlaps = other_lower - eps < low < other_upper - eps
                    up_overlaps = other_lower + eps < up < other_upper + eps
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
        spaces (): :py:class:`~zfit.Space`s to extract the obs from

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


def limits_consistent(spaces: Iterable["zfit.Space"]):
    """Check if space limits are the *exact* same in each obs they are defined and therefore are compatible.

    In this case, if a space has several limits, e.g. from -1 to 1 and from 2 to 3 (all in the same observable),
    to be consistent with this limits, other limits have to have (in this obs) also the limits
    from -1 to 1 and from 2 to 3. Only having the limit -1 to 1 _or_ 2 to 3 is considered _not_ consistent.

    This function is useful to check if several spaces with *different* observables can be _combined_.

    Args:
        spaces (List[zfit.Space]):

    Returns:
        bool:
    """
    try:
        new_space = combine_spaces(spaces=spaces)
    except LimitsIncompatibleError:
        return False
    return bool(new_space)


def combine_spaces(spaces: Iterable["zfit.Space"]):
    """Combine spaces with different `obs` and `limits` to one `space`.

    Checks if the limits in each obs coincide *exactly*. If this is not the case, the combination
    is not unambiguous and `False` is returned

    Args:
        spaces (List[:py:class:`~zfit.Space`]):

    Returns:
        `zfit.Space` or False: Returns False if the limits don't coincide in one or more obs. Otherwise
            return the :py:class:`~zfit.Space` with all obs from `spaces` sorted by the order of `spaces` and with the
            combined limits.
    Raises:
        ValueError: if only one space is given
        LimitsIncompatibleError: If the limits of one or more spaces (or within a space) overlap
        LimitsNotSpecifiedError: If the limits for one or more obs but not all are None.
    """
    spaces = convert_to_container(spaces, container=tuple)
    # if len(spaces) <= 1:
    #     return spaces
    # raise ValueError("Need at least two spaces to test limit consistency.")  # TODO: allow? usecase?

    all_obs = common_obs(spaces=spaces)
    all_lower = []
    all_upper = []
    spaces = tuple(space.with_obs(all_obs) for space in spaces)

    # create the lower and upper limits with all obs replacing missing dims with None
    # With this, all limits have the same length
    if limits_overlap(spaces=spaces, allow_exact_match=True):
        raise LimitsIncompatibleError("Limits overlap")

    for space in spaces:
        if space.limits is None:
            continue
        lowers, uppers = space.limits
        lower = [tuple(low[space.obs.index(ob)] for low in lowers) if ob in space.obs else None for ob in all_obs]
        upper = [tuple(up[space.obs.index(ob)] for up in uppers) if ob in space.obs else None for ob in all_obs]
        all_lower.append(lower)
        all_upper.append(upper)

    def check_extract_limits(limits_spaces):
        new_limits = []

        if not limits_spaces:
            return None
        for index, obs in enumerate(all_obs):
            current_limit = None
            for limit in limits_spaces:
                lim = limit[index]

                if lim is not None:
                    if current_limit is None:
                        current_limit = lim
                    elif not np.allclose(current_limit, lim):
                        return False
            else:
                if current_limit is None:
                    raise LimitsNotSpecifiedError("Limits in obs {} are not specified".format(obs))
                new_limits.append(current_limit)

        n_limits = int(np.prod(tuple(len(lim) for lim in new_limits)))
        new_limits_comb = [[] for _ in range(n_limits)]
        for limit in new_limits:
            for lim in limit:
                for i in range(int(n_limits / len(limit))):
                    new_limits_comb[i].append(lim)

        new_limits = tuple(tuple(limit) for limit in new_limits_comb)
        return new_limits

    new_lower = check_extract_limits(all_lower)
    new_upper = check_extract_limits(all_upper)
    assert not (new_lower is None) ^ (new_upper is None), "Bug, please report issue. either both are defined or None."
    if new_lower is None:
        limits = None
    elif new_lower is False:
        return False
    else:
        limits = (new_lower, new_upper)
    new_space = zfit.Space(obs=all_obs, limits=limits)
    return new_space

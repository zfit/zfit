#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .interfaces import ZfitDimensional
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import SpaceIncompatibleError


class BaseDimensional(ZfitDimensional):
    def _check_n_obs(self, space):
        if self._N_OBS is not None:
            if len(space.obs) != self._N_OBS:
                raise SpaceIncompatibleError(
                    f"Exactly {self._N_OBS} obs are allowed, {space.obs} are given."
                )

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


def get_same_obs(obs):
    deps = [set() for _ in range(len(obs))]
    for i, ob in enumerate(obs):
        for j, other_ob in enumerate(obs[i + 1 :]):
            if not set(ob).isdisjoint(other_ob):
                deps[i].add(i)
                deps[i].add(j + i + 1)
                deps[j + i + 1].add(i)

    deps = tuple(tuple(dep) for dep in deps)

    return deps


def limits_overlap(
    spaces: ztyping.SpaceOrSpacesTypeInput, allow_exact_match: bool = False
) -> bool:
    """Check if _any_ of the limits of ``spaces`` overlaps with _any_ other of ``spaces``.

    This also checks multiple limits within one space. If ``allow_exact_match`` is set to true, then
    an *exact* overlap of limits is allowed.


    Args:
        spaces:
        allow_exact_match: An exact overlap of two limits is counted as "not overlapping".
            Example: limits from -1 to 3 and 4 to 5 to *NOT* overlap with the limits 4 to 5 *iff*
            ``allow_exact_match`` is True.

    Returns:
        If there are overlapping limits.
    """
    # TODO(Mayou36): add approx comparison global in zfit
    eps = 1e-8  # epsilon for float comparisons
    spaces = convert_to_container(spaces, container=tuple)
    all_obs = common_obs(spaces=spaces)
    for obs in all_obs:
        lowers = []
        uppers = []
        for space in spaces:
            if not space.has_limits or obs not in space.obs:
                continue
            else:
                index = space.obs.index(obs)

            for spa in space:
                lower, upper = spa.rect_limits  # TODO: new space
                low = lower[:, index]
                up = upper[:, index]

                for other_lower, other_upper in zip(lowers, uppers):
                    if (
                        allow_exact_match
                        and np.allclose(other_lower, low)
                        and np.allclose(other_upper, up)
                    ):
                        continue
                    # TODO(Mayou36): tol? add global flags?
                    low_overlaps = np.all(other_lower - eps < low) and np.all(
                        low < other_upper - eps
                    )
                    up_overlaps = np.all(other_lower + eps < up) and np.all(
                        up < other_upper + eps
                    )
                    overlap = low_overlaps or up_overlaps
                    if overlap:
                        return True
                lowers.append(low)
                uppers.append(up)
    return False


def common_obs(spaces: ztyping.SpaceOrSpacesTypeInput) -> list[str] | bool:
    """Extract the union of ``obs`` from ``spaces`` in the order of ``spaces``.

    For example:
        | space1.obs: ['obs1', 'obs3']
        | space2.obs: ['obs2', 'obs3', 'obs1']
        | space3.obs: ['obs2']

        returns ['obs1', 'obs3', 'obs2']

    Args:
        spaces: :py:class:`~zfit.Space`s to extract the obs from

    Returns:
        The observables as ``str`` or False if not every space has observables
    """
    spaces = convert_to_container(spaces, container=tuple)
    all_obs = []
    for space in spaces:
        if space.obs is None:
            return False
        for ob in space.obs:
            if ob not in all_obs:
                all_obs.append(ob)
    return all_obs


def common_axes(spaces: ztyping.SpaceOrSpacesTypeInput) -> list[str] | bool:
    """Extract the union of ``axes`` from ``spaces`` in the order of ``spaces``.

    For example:
        | space1.axes: [1, 3]
        | space2.axes: [2, 3, 1]
        | space3.axes: [2]

        returns [1, 3, 2]

    Args:
        spaces: :py:class:`~zfit.Space`s to extract the axes from

    Returns:
        The axes as int or False if not every space has axes
    """
    spaces = convert_to_container(spaces, container=tuple)
    all_axes = []
    for space in spaces:
        if space.axes is None:
            return False
        for ax in space.axes:
            if ax not in all_axes:
                all_axes.append(ax)
    return all_axes


def obs_subsets(
    dimensionals: Iterable[ZfitDimensional],
) -> dict[set[str], ZfitDimensional]:
    """Split ``dimensionals`` into the smallest subgroup of obs and return a dict.

    Args:
        dimensionals: An Iterable containing two or more ZfitDimensional that should be split into the smallest subset.

    Returns:
        Dict with the keys being sets of observables and the values, an iterable, containing the ZfitDimensional
    """
    obs_dims = {}
    for dim in dimensionals:
        for obs in obs_dims:
            if obs.intersection(dim.obs):
                union = obs.union(dim.obs)
                obs_dims[union] = obs_dims.pop(obs)
                obs_dims[union].append(dim)
                break  # we had a match, go to the next dim
        else:
            obs_dims[frozenset(dim.obs)] = [dim]

    return obs_dims

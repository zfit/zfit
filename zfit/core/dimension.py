import functools

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

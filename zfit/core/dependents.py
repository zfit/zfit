#  Copyright (c) 2022 zfit

from __future__ import annotations

#  Copyright (c) 2020 zfit
import abc
import itertools
from collections.abc import Iterable

from ordered_set import OrderedSet

from zfit.core.interfaces import ZfitDependenciesMixin, ZfitObject
from zfit.util import ztyping
from zfit.util.container import convert_to_container


class BaseDependentsMixin(ZfitDependenciesMixin):
    @abc.abstractmethod
    def _get_dependencies(self) -> ztyping.DependentsType:
        raise NotImplementedError

    def get_cache_deps(self, only_floating: bool = True) -> ztyping.DependentsType:
        """Return a set of all independent :py:class:`~zfit.Parameter` that this object depends on.

        Args:
            only_floating: If ``True``, only return floating :py:class:`~zfit.Parameter`
        """
        dependencies = self._get_dependencies()
        if only_floating:
            dependencies = OrderedSet(filter(lambda p: p.floating, dependencies))
        return dependencies


def _extract_dependencies(zfit_objects: Iterable[ZfitObject]) -> ztyping.DependentsType:
    """Calls the :py:meth:`~BaseDependentsMixin.get_dependents` method on every object and returns a combined set.

    Args:
        zfit_objects:

    Returns:
        A set of independent Parameters
    """
    zfit_objects = convert_to_container(zfit_objects)
    dependents = (obj.get_cache_deps(only_floating=False) for obj in zfit_objects)
    dependents_set = OrderedSet(itertools.chain.from_iterable(dependents))  # flatten
    return dependents_set

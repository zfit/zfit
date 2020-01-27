#  Copyright (c) 2020 zfit
import abc
import itertools
from typing import Iterable

from ordered_set import OrderedSet

from zfit.core.interfaces import ZfitDependentsMixin, ZfitObject
from zfit.util import ztyping
from zfit.util.container import convert_to_container


class BaseDependentsMixin(ZfitDependentsMixin):
    @abc.abstractmethod
    def _get_dependents(self) -> ztyping.DependentsType:
        raise NotImplementedError

    def get_dependents(self, only_floating: bool = True) -> ztyping.DependentsType:
        """Return a set of all independent :py:class:`~zfit.Parameter` that this object depends on.

        Args:
            only_floating (bool): If `True`, only return floating :py:class:`~zfit.Parameter`
        """
        dependents = self._get_dependents()
        if only_floating:
            dependents = OrderedSet(filter(lambda p: p.floating, dependents))
        return dependents

    @staticmethod
    def _extract_dependents(zfit_objects: Iterable[ZfitObject]) -> ztyping.DependentsType:
        """Calls the :py:meth:`~BaseDependentsMixin.get_dependents` method on every object and returns a combined set.

        Args:
            zfit_objects ():

        Returns:
            set(zfit.Parameter): A set of independent Parameters
        """
        zfit_objects = convert_to_container(zfit_objects)
        dependents = (obj.get_dependents(only_floating=False) for obj in zfit_objects)
        dependents_set = OrderedSet(itertools.chain.from_iterable(dependents))  # flatten
        return dependents_set

import abc
import itertools

from .interfaces import ZfitObject
from ..util.container import convert_to_container


class BaseObject(ZfitObject):

    def get_dependents(self, only_floating: bool = False):
        """Return a list of all independent :py:class:`~zfit.Parameter` that this object depends on.

        Args:
            only_floating (bool): If `True`, only return floating :py:class:`~zfit.Parameter`
        """
        dependents = self._get_dependents()
        if only_floating:
            dependents = set(filter(lambda p: p.floating, dependents))
        return dependents

    @abc.abstractmethod
    def _get_dependents(self):
        raise NotImplementedError

    @staticmethod
    def _extract_dependents(zfit_objects):
        zfit_object = convert_to_container(zfit_objects)
        dependents = (obj.get_dependents() for obj in zfit_object)
        dependents = set(itertools.chain.from_iterable(dependents))  # flatten
        return dependents

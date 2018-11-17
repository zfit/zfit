import abc
import itertools

import pep487

from zfit.util.container import convert_to_container


class ZfitObject(pep487.ABC):

    @abc.abstractmethod
    def get_dependents(self, only_floating=False):
        raise NotImplementedError


class BaseObject(ZfitObject):

    def get_dependents(self, only_floating: bool = False):
        """Return a list of all independent :py:class:`~zfit.Parameter` that this object depends on.

        Args:
            only_floating (bool): If `True`, only return floating :py:class:`~zfit.Parameter`
        """
        return self._get_dependents(only_floating=only_floating)

    @abc.abstractmethod
    def _get_dependents(self, only_floating):
        raise NotImplementedError

    @staticmethod
    def _extract_dependents(zfit_objects, only_floating):
        zfit_object = convert_to_container(zfit_objects)
        dependents = (obj.get_dependents(only_floating=only_floating) for obj in zfit_object)
        dependents = set(itertools.chain.from_iterable(dependents))  # flatten
        return dependents

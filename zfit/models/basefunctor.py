import abc
from typing import List

from zfit.core.interfaces import ZfitFunctorMixin, ZfitModel


class FunctorMixin(ZfitFunctorMixin):

    def _get_dependents(self):
        dependents = super()._get_dependents()  # get the own parameter dependents
        model_dependents = self._extract_dependents(self.models)
        return dependents.union(model_dependents)

    @property
    def models(self) -> List[ZfitModel]:
        """Return the models of this `Functor`. Can be `pdfs` or `funcs`."""
        return self._models

    @property
    @abc.abstractmethod
    def _models(self) -> List[ZfitModel]:
        raise NotImplementedError

    def dims(self):
        pass





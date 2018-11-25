import abc
from typing import List, Union, Dict

from zfit.core.dimension import get_same_dims
from zfit.core.interfaces import ZfitFunctorMixin, ZfitModel
from zfit.util.exception import DimsNotUnambigiousError


class FunctorMixin(ZfitFunctorMixin):

    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)
        if self.dims is None:
            proposed_dims = set(model.dims for model in models)
            if len(proposed_dims) == 1:
                self.dims = models[0].dims
            else:
                raise DimsNotUnambigiousError("Dimensions are `None` and cannot be taken from the"
                                              "models, as their dimensions are not the same.")
        self._model_dims = tuple(self._check_convert_input_dims(model.dims) for model in models)

    def _get_dependents(self):
        dependents = super()._get_dependents()  # get the own parameter dependents
        model_dependents = self._extract_dependents(self.get_models())
        return dependents.union(model_dependents)

    @property
    def models(self) -> Dict[Union[float, int, str], ZfitModel]:
        """Return the models of this `Functor`. Can be `pdfs` or `funcs`."""
        return self._models

    @property
    def _model_same_dims(self):  # TODO(mayou36): cache?
        return get_same_dims(self._model_dims)

    @property
    @abc.abstractmethod
    def _models(self) -> Dict[Union[float, int, str], ZfitModel]:
        raise NotImplementedError

    def get_models(self, names=None) -> List[ZfitModel]:
        if names is None:
            models = list(self.models.values())
        else:
            models = [self.models[name] for name in names]
        return models

    @property
    def _n_dims(self):
        return len(self.dims)

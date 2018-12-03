import abc
from typing import List, Union, Dict

from zfit.core.basemodel import BaseModel
from zfit.core.dimension import get_same_dims
from zfit.core.interfaces import ZfitFunctorMixin, ZfitModel
from zfit.util.container import convert_to_container
from zfit.util.exception import AxesNotUnambiguousError


class FunctorMixin(ZfitFunctorMixin, BaseModel):

    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)
        models = convert_to_container(models, container=list)
        self._model_dims_index = self._check_convert_model_dims_to_index(models=models)

    def _check_convert_model_dims_to_index(self, models):
        models_dims_index = None
        models_dims = tuple(model.dims for model in models)
        if self.dims is None:
            # try to infer from the models
            proposed_dims = set(models_dims)
            if len(proposed_dims) == 1:  # if all submodels have the *exact* same dims -> intention is "clear"
                proposed_dim = proposed_dims.pop()

                # models_dims are None and functor dims is None -> allow for easy use-case of sum(exp, gauss)
                if proposed_dim is None and not self._functor_allow_none_dims:
                    raise AxesNotUnambiguousError("Dims of submodels as well as functor are None."
                                                  "Not allowed for this functor. Specify the dims in the"
                                                  "submodels and/or in the Functor.")
                # in this case, at least the n_dims should coincide
                elif proposed_dim is None:
                    models_n_dims = set(model.n_dims for model in models)
                    if len(models_n_dims) == 1:
                        models_dims_index = tuple(range(len(models_n_dims))) * len(models)
                    else:
                        raise AxesNotUnambiguousError("n_dims of models are different and dims are all `None`. "
                                                      "Therefore they can't be inferered safely. Either use same ranked"
                                                      "models or specify explicitely the dims.")

                self.dims = proposed_dim

            # different dimensions in submodels -> how to merge? Ambiguous
            else:
                raise AxesNotUnambiguousError("Dimensions are `None` for this functor and cannot be taken from the"
                                              "models, as their dimensions are not *exactly* the same.")
        if models_dims_index is None:
            try:
                models_dims_index = tuple(tuple(self.dims.index(dim) for dim in dims) for dims in models_dims )
            except ValueError:
                missing_dims = set(models_dims) - set(self.dims)
                raise ValueError("The following dims are not specified in the pdf: {}".format(str(missing_dims)))

        return models_dims_index

    @property
    @abc.abstractmethod
    def _functor_allow_none_dims(self) -> bool:
        """If True, allow to set the dims to None. Otherwise raise a `AxesNotUnambiguousError`.

        Returns:
            bool:
        """
        raise NotImplementedError

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
        return get_same_dims(self._model_dims_index)

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
        if self.dims is None:
            return None
        else:
            return len(self.dims)

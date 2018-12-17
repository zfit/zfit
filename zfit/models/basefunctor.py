import abc
from typing import List, Union, Dict, Tuple

from zfit.core.limits import NamedSpace
from zfit.core.basemodel import BaseModel
from zfit.core.dimension import get_same_dims, get_same_obs
from zfit.core.interfaces import ZfitFunctorMixin, ZfitModel
from zfit.util.container import convert_to_container
from zfit.util.exception import AxesNotUnambiguousError, NormRangeNotSpecifiedError


class FunctorMixin(ZfitFunctorMixin, BaseModel):

    def __init__(self, models, obs, **kwargs):
        models = convert_to_container(models, container=list)
        obs = self._check_extract_input_obs(obs=obs, models=models)

        super().__init__(obs=obs, **kwargs)
        self._model_obs = tuple(model.obs for model in models)

    def _check_extract_input_obs(self, obs, models):
        extracted_obs = _extract_common_obs(obs=tuple(model.obs for model in models))
        if obs is None:
            obs = extracted_obs
        else:
            if isinstance(obs, NamedSpace):
                obs_str = obs.obs
            else:
                obs_str = convert_to_container(value=obs, container=tuple)
            if not frozenset(obs_str) == frozenset(extracted_obs):
                raise ValueError("The given obs do not coincide with the obs from the daughter models.")
        return obs

    # TODO(Mayou36): implement properly with obs

    def _check_convert_model_dims_to_index(self, models):
        models_dims_index = None
        models_dims = tuple(model.axes for model in models)
        if self.axes is None:
            # try to infer from the models
            proposed_dims = set(models_dims)
            if len(proposed_dims) == 1:  # if all submodels have the *exact* same axes -> intention is "clear"
                proposed_dim = proposed_dims.pop()

                # models_dims are None and functor axes is None -> allow for easy use-case of sum(exp, gauss)
                if proposed_dim is None and not self._functor_allow_none_dims:
                    raise AxesNotUnambiguousError("Dims of submodels as well as functor are None."
                                                  "Not allowed for this functor. Specify the axes in the"
                                                  "submodels and/or in the Functor.")
                # in this case, at least the n_obs should coincide
                elif proposed_dim is None:
                    models_n_dims = set(model.n_obs for model in models)
                    if len(models_n_dims) == 1:
                        models_dims_index = tuple(range(len(models_n_dims))) * len(models)
                    else:
                        raise AxesNotUnambiguousError("n_obs of models are different and axes are all `None`. "
                                                      "Therefore they can't be inferered safely. Either use same ranked"
                                                      "models or specify explicitely the axes.")

                self.obs = proposed_dim

            # different dimensions in submodels -> how to merge? Ambiguous
            else:
                raise AxesNotUnambiguousError("Dimensions are `None` for this functor and cannot be taken from the"
                                              "models, as their dimensions are not *exactly* the same.")
        if models_dims_index is None:
            try:
                models_dims_index = tuple(tuple(self.axes.index(dim) for dim in dims) for dims in models_dims)
            except ValueError:
                missing_dims = set(models_dims) - set(self.axes)
                raise ValueError("The following axes are not specified in the pdf: {}".format(str(missing_dims)))

        return models_dims_index

    def _get_dependents(self):
        dependents = super()._get_dependents()  # get the own parameter dependents
        model_dependents = self._extract_dependents(self.get_models())
        return dependents.union(model_dependents)

    @property
    def models(self) -> Dict[Union[float, int, str], ZfitModel]:
        """Return the models of this `Functor`. Can be `pdfs` or `funcs`."""
        return self._models

    @property
    def _model_same_obs(self):
        return get_same_obs(self._model_obs)

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
        if self.axes is None:
            return None
        else:
            return len(self.axes)

    def _check_input_norm_range_default(self, norm_range, caller_name="", none_is_error=True):
        if norm_range is None:
            try:
                norm_range = self.norm_range
            except AttributeError:
                raise NormRangeNotSpecifiedError("The normalization range is `None`, no default norm_range is set")
        return self._check_input_norm_range(norm_range=norm_range, caller_name=caller_name, none_is_error=none_is_error)


def _extract_common_obs(obs: Tuple[Union[Tuple[str], NamedSpace]]) -> Tuple[str]:
    obs_iter = [space.obs if isinstance(space, NamedSpace) else space for space in obs]
    unique_obs = []
    for obs in obs_iter:
        for o in obs:
            if o not in unique_obs:
                unique_obs.append(o)
    return tuple(unique_obs)

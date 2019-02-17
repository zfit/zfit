import abc
from typing import List, Union, Dict, Tuple

import zfit
from ..core.limits import Space
from ..core.basemodel import BaseModel
from ..core.dimension import get_same_obs, combine_spaces
from ..core.interfaces import ZfitFunctorMixin, ZfitModel
from ..util.container import convert_to_container
from ..util.exception import AxesNotUnambiguousError, NormRangeNotSpecifiedError, LimitsIncompatibleError


class FunctorMixin(ZfitFunctorMixin, BaseModel):

    def __init__(self, models, obs, **kwargs):
        models = convert_to_container(models, container=list)
        obs = self._check_extract_input_obs(obs=obs, models=models)

        super().__init__(obs=obs, **kwargs)
        self._model_obs = tuple(model.obs for model in models)

    def _infer_obs_from_daughters(self):
        obs = set(self._model_obs)
        if len(obs) == 1:
            return obs.pop()
        else:
            return False

    def _check_extract_input_obs(self, obs, models):

        # combine spaces and limits
        try:
            models_space = combine_spaces([model.space for model in models])
        except LimitsIncompatibleError:  # then only add obs
            extracted_obs = _extract_common_obs(obs=tuple(model.obs for model in models))
            models_space = Space(obs=extracted_obs)

        if obs is None:
            obs = models_space
        else:
            if isinstance(obs, Space):
                obs_str = obs.obs
            else:
                obs_str = convert_to_container(value=obs, container=tuple)
            # if not frozenset(obs_str) == frozenset(models_space.obs):  # not needed, example projection
            #     raise ValueError("The given obs do not coincide with the obs from the daughter models.")
        return obs

    def _get_dependents(self):
        dependents = super()._get_dependents()  # get the own parameter dependents
        model_dependents = self._extract_dependents(self.get_models())
        return dependents.union(model_dependents)

    @property
    def models(self) -> List[ZfitModel]:
        """Return the models of this `Functor`. Can be `pdfs` or `funcs`."""
        return self._models

    @property
    def _model_same_obs(self):
        return get_same_obs(self._model_obs)

    @property
    @abc.abstractmethod
    def _models(self) -> List[ZfitModel]:
        raise NotImplementedError

    def get_models(self, names=None) -> List[ZfitModel]:
        if names is None:
            models = list(self.models)
        else:
            raise ValueError("name not supported currently.")
            # models = [self.models[name] for name in names]
        return models

    def _check_input_norm_range_default(self, norm_range, caller_name="", none_is_error=True):
        if norm_range is None:
            try:
                norm_range = self.norm_range
            except AttributeError:
                raise NormRangeNotSpecifiedError("The normalization range is `None`, no default norm_range is set")
        return self._check_input_norm_range(norm_range=norm_range, caller_name=caller_name, none_is_error=none_is_error)


def _extract_common_obs(obs: Tuple[Union[Tuple[str], Space]]) -> Tuple[str]:
    obs_iter = [space.obs if isinstance(space, Space) else space for space in obs]
    unique_obs = []
    for obs in obs_iter:
        for o in obs:
            if o not in unique_obs:
                unique_obs.append(o)
    return tuple(unique_obs)

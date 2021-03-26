#  Copyright (c) 2021 zfit

import abc
from typing import Iterable, List, Optional, Set, Tuple, Union

from ordered_set import OrderedSet

from ..core.basemodel import BaseModel
from ..core.dependents import _extract_dependencies
from ..core.dimension import get_same_obs
from ..core.interfaces import ZfitFunctorMixin, ZfitModel, ZfitSpace
from ..core.space import Space, combine_spaces
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (LimitsIncompatibleError,
                              NormRangeNotSpecifiedError,
                              SpaceIncompatibleError)


def extract_daughter_input_obs(obs: ztyping.ObsTypeInput, spaces: Iterable[ZfitSpace]) -> ZfitSpace:
    """Extract the common space from `spaces` by combining them, test against obs.

    The `obs` are assumed to be the obs given to a functor while the `spaces` are the spaces of the daughters.
    First, the combined space from the daughters is extracted. If no `obs` are given, this is returned.
    If `obs` are given, it is checked whether they agree. If they agree, and no limit is set on `obs` (i.e. they
    are pure strings), the inferred limits are used, sorted by obs. Otherwise, obs is directly used.

    Args:
        obs:
        spaces:

    Returns:
    """
    spaces = convert_to_container(spaces)
    # combine spaces and limits
    try:
        models_space = combine_spaces(*spaces)
    except LimitsIncompatibleError:  # then only add obs
        extracted_obs = _extract_common_obs(obs=tuple(space.obs for space in spaces))
        models_space = Space(obs=extracted_obs)

    if obs is None:
        obs = models_space
    else:
        if isinstance(obs, Space):
            obs = obs
        else:
            obs = Space(obs=obs)
        # if not frozenset(obs.obs) == frozenset(models_space.obs):  # not needed, example projection
        #     raise SpaceIncompatibleError("The given obs do not coincide with the obs from the daughter models.")
        if not obs.obs == models_space.obs and not obs.limits_are_set:
            obs = models_space.with_obs(obs.obs)

    return obs


class FunctorMixin(ZfitFunctorMixin, BaseModel):

    def __init__(self, models, obs, **kwargs):
        models = convert_to_container(models, container=list)
        obs = extract_daughter_input_obs(obs=obs, spaces=[model.space for model in models])

        super().__init__(obs=obs, **kwargs)
        # TODO: needed? remove below
        self._model_obs = tuple(model.obs for model in models)

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = super()._get_params(floating, is_yield, extract_independent)
        if is_yield is not True:
            params = params.union(*(model.get_params(floating=floating, is_yield=False,
                                                     extract_independent=extract_independent)
                                    for model in self.models))
        return params

    # def _infer_space_from_daughters(self):
    #     space = set(model.space for model in self.models)
    #     obs = set(norm_range.obs for norm_range in space)
    #     if len(space) == 1:
    #         return space.pop()
    #     elif len(obs) > 1:  # TODO(Mayou36, #77): different obs?
    #         return None
    #     else:
    #         return False

    def _get_dependencies(self):
        dependents = super()._get_dependencies()  # get the own parameter dependents
        model_dependents = _extract_dependencies(self.get_models())
        return dependents.union(model_dependents)

    @property
    def models(self) -> List[ZfitModel]:
        """Return the models of this `Functor`.

        Can be `pdfs` or `funcs`.
        """
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
        return self._check_input_norm_range(norm_range=norm_range, none_is_error=none_is_error)


def _extract_common_obs(obs: Tuple[Union[Tuple[str], Space]]) -> Tuple[str]:
    obs_iter = [space.obs if isinstance(space, Space) else space for space in obs]
    unique_obs = []
    for obs in obs_iter:
        for o in obs:
            if o not in unique_obs:
                unique_obs.append(o)
    return tuple(unique_obs)

#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import List, Optional

import pydantic
import tensorflow as tf

from ..core.coordinates import convert_to_obs_str
from ..core.dependents import _extract_dependencies
from ..core.dimension import get_same_obs
from ..core.interfaces import ZfitFunctorMixin, ZfitModel, ZfitSpace, ZfitParameter
from ..core.parameter import convert_to_parameter
from ..core.space import Space, combine_spaces
from ..serialization import SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..serialization.serializer import Serializer
from ..settings import ztypes, run
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_norm_range
from ..util.exception import (
    LimitsIncompatibleError,
    NormRangeNotSpecifiedError,
    ObsIncompatibleError,
    ModelIncompatibleError,
)
from ..util.warnings import warn_advanced_feature, warn_changed_feature
from ..z import numpy as znp


def extract_daughter_input_obs(
    obs: ztyping.ObsTypeInput, spaces: Iterable[ZfitSpace]
) -> ZfitSpace:
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


class FunctorMixin(ZfitFunctorMixin):
    def __init__(self, models, obs, **kwargs):
        models = convert_to_container(models, container=list)
        obs = extract_daughter_input_obs(
            obs=obs, spaces=[model.space for model in models]
        )

        super().__init__(obs=obs, **kwargs)
        # TODO: needed? remove below
        self._model_obs = tuple(model.obs for model in models)
        self._models = models

    def _get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
    ) -> set[ZfitParameter]:
        params = super()._get_params(floating, is_yield, extract_independent)
        if is_yield is not True:
            params = params.union(
                *(
                    model.get_params(
                        floating=floating,
                        is_yield=False,
                        extract_independent=extract_independent,
                    )
                    for model in self.models
                )
            )
        return params

    def _get_dependencies(self):
        dependents = super()._get_dependencies()  # get the own parameter dependents
        model_dependents = _extract_dependencies(self.get_models())
        return dependents.union(model_dependents)

    @property
    def models(self) -> list[ZfitModel]:
        """Return the models of this `Functor`.

        Can be `pdfs` or `funcs`.
        """
        return list(self._models)

    @property
    def _model_same_obs(self):
        return get_same_obs(self._model_obs)

    def get_models(self, names=None) -> list[ZfitModel]:
        if names is None:
            models = list(self.models)
        else:
            raise ValueError("name not supported currently.")
            # models = [self.models[name] for name in names]
        return models

    @deprecated_norm_range
    def _check_input_norm_default(self, norm, caller_name="", none_is_error=True):
        if norm is None:
            try:
                norm = self.norm_range
            except AttributeError:
                raise NormRangeNotSpecifiedError(
                    "The normalization range is `None`, no default norm is set"
                )
        return self._check_input_norm_range(norm=norm, none_is_error=none_is_error)


class FunctorPDFRepr(BasePDFRepr):
    _implementation = None
    pdfs: List[Serializer.types.PDFTypeDiscriminated]
    obs: Optional[SpaceRepr] = None

    @pydantic.root_validator(pre=True)
    def validate_all_functor(cls, values):
        if cls.orm_mode(values):
            init = values["hs3"].original_init
            values = dict(values)
            values["obs"] = init["obs"]
            values["extended"] = init["extended"]
        return values


def _extract_common_obs(obs: tuple[tuple[str] | Space]) -> tuple[str]:
    obs_iter = [space.obs if isinstance(space, Space) else space for space in obs]
    unique_obs = []
    for obs in obs_iter:
        for o in obs:
            if o not in unique_obs:
                unique_obs.append(o)
    return tuple(unique_obs)


def _preprocess_init_sum(fracs, obs, pdfs):
    frac_param_created = False
    if len(pdfs) < 2:
        raise ValueError(f"Cannot build a sum of less than two pdfs {pdfs}")
    common_obs = obs if obs is not None else pdfs[0].obs
    common_obs = convert_to_obs_str(common_obs)
    if any(frozenset(pdf.obs) != frozenset(common_obs) for pdf in pdfs):
        raise ObsIncompatibleError(
            "Currently, sums are only supported in the same observables"
        )
    # check if all extended
    are_extended = [pdf.is_extended for pdf in pdfs]
    all_extended = all(are_extended)
    no_extended = not any(are_extended)
    fracs = convert_to_container(fracs)
    if fracs:  # not None or empty list
        fracs = [convert_to_parameter(frac) for frac in fracs]
    elif not all_extended:
        raise ModelIncompatibleError(
            f"Not all pdf {pdfs} are extended and no fracs {fracs} are provided."
        )
    if not no_extended and fracs:
        warn_advanced_feature(
            f"This SumPDF is built with fracs {fracs} and {'all' if all_extended else 'some'} "
            f"pdf are extended: {pdfs}."
            f" This will ignore the yields of the already extended pdfs and the result will"
            f" be a not extended SumPDF.",
            identifier="sum_extended_frac",
        )
    # catch if args don't fit known case
    if fracs:
        # create fracs if one is missing
        if len(fracs) == len(pdfs) - 1:
            frac_param_created = True
            frac_params_tmp = {f"frac_{i}": frac for i, frac in enumerate(fracs)}

            def remaining_frac_func(params):
                return tf.constant(1.0, dtype=ztypes.float) - tf.add_n(
                    list(params.values())
                )

            remaining_frac = convert_to_parameter(
                remaining_frac_func, params=frac_params_tmp
            )
            if run.numeric_checks:
                tf.debugging.assert_non_negative(
                    remaining_frac,
                    f"The remaining fraction is negative, the sum of fracs is > 0. Fracs: {fracs}",
                )  # check fractions

            # IMPORTANT to change the name! Otherwise, recursion due to namespace capture in the lambda
            fracs_cleaned = fracs + [remaining_frac]

        elif len(fracs) == len(pdfs):
            warn_changed_feature(
                "A SumPDF with the number of fractions equal to the number of pdf will no longer "
                "be extended. To make it extended, either manually use 'create_exteneded' or set "
                "the yield. OR provide all pdfs as extended pdfs and do not provide a fracs "
                "argument.",
                identifier="new_sum",
            )
            fracs_cleaned = fracs

        else:
            raise ModelIncompatibleError(
                f"If all PDFs are not extended {pdfs}, the fracs {fracs} have to be of"
                f" the same length as pdf or one less."
            )
        param_fracs = fracs_cleaned
    # for the extended case, take the yields, normalize them, in case no fracs are given.
    sum_yields = None
    if all_extended and not fracs:
        yields = [pdf.get_yield() for pdf in pdfs]

        def sum_yields_func(params):
            return znp.sum(list(params.values()))

        sum_yields = convert_to_parameter(
            sum_yields_func, params={f"yield_{i}": y for i, y in enumerate(yields)}
        )
        yield_fracs = [
            convert_to_parameter(
                lambda params: params["yield_"] / params["sum_yields"],
                params={"sum_yields": sum_yields, "yield_": yield_},
            )
            for yield_ in yields
        ]

        fracs_cleaned = None
        param_fracs = yield_fracs
    params = OrderedDict()
    for i, frac in enumerate(param_fracs):
        params[f"frac_{i}"] = frac
    return (
        all_extended,
        fracs_cleaned,
        param_fracs,
        params,
        sum_yields,
        frac_param_created,
    )

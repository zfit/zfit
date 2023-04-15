#  Copyright (c) 2023 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

from ..util.ztyping import ExtendedInputType, NormInputType

if TYPE_CHECKING:
    pass

from collections.abc import Mapping

import functools
import warnings

import tensorflow as tf

import zfit.z.numpy as znp

from .. import z
from ..core.interfaces import (
    ZfitIndependentParameter,
    ZfitPDF,
    ZfitSpace,
    ZfitParameter,
)
from ..core.basepdf import BasePDF
from ..core.parameter import set_values
from ..core.space import combine_spaces, convert_to_space, supports
from ..util.exception import WorkInProgressError
from ..util.warnings import warn_experimental_feature
from .functor import BaseFunctor


class ConditionalPDFV1(BaseFunctor):
    @warn_experimental_feature
    def __init__(
        self,
        pdf: ZfitPDF,
        cond: Mapping[ZfitIndependentParameter, ZfitSpace],
        *,
        name: str = "ConditionalPDF",
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        use_vectorized_map: bool = False,
        sample_with_replacement: bool = True,
    ) -> None:
        """EXPERIMENTAL! Implementation of a Conditional PDF, rather slow and for research purpose.

        As an example, a Gaussian is wrapped in order to make 'sigma' conditional.

        .. jupyter-execute::



        Args:
            pdf: PDF that will be wrapped. Convert one or several parameters of *pdf* to a conditional
                parameter, meaning that the parameter *param* in the ``cond`` mapping will now be
                determined by the data in the ``Space``, the value of the ``cond``.
            cond: Mapping of parameter to input data.
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:model.init.norm| The normalization of the PDF. |@docend:model.init.norm|
            use_vectorized_map ():
            sample_with_replacement ():
        """
        # TODO: add to serializer, see below repr for problem
        # original_init = {'pdf': pdf, 'cond': cond, 'name': name, 'extended': extended, 'norm': norm,
        #                  'use_vectorized_map': use_vectorized_map, 'sample_with_replacement': sample_with_replacement}
        self._sample_with_replacement = sample_with_replacement
        self._use_vectorized_map = use_vectorized_map
        self._cond, cond_obs = self._check_input_cond(cond)
        obs = pdf.space * cond_obs
        super().__init__(pdfs=pdf, obs=obs, name=name, extended=extended, norm=norm)
        # self.hs3.original_init.update(original_init)  # TODO: add to serializer

    @property
    def cond(self) -> dict[ZfitIndependentParameter, ZfitSpace]:
        return self._cond

    def _check_input_cond(self, cond):
        spaces = []
        for param, obs in cond.items():
            if not isinstance(param, ZfitIndependentParameter):
                raise TypeError(f"parameter {param} not a ZfitIndependentParameter")
            spaces.append(convert_to_space(obs))
        return cond, combine_spaces(*spaces)

    @supports(norm=True, multiple_limits=True)
    @z.function(wraps="conditional_pdf")
    def _pdf(self, x, norm):
        pdf = self.pdfs[0]
        param_x_indices = {
            p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()
        }
        x_values = x.value()

        if self._use_vectorized_map:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(
                shape=(1, *x_values.shape[1:-1]), dtype=self.dtype
            )
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)

        # TODO: reset parameters?

        def eval_pdf(cond_and_data):
            x_pdf = cond_and_data[None, ..., : pdf.n_obs]
            for param, index in param_x_indices.items():
                param.assign(cond_and_data[..., index])
            return pdf.pdf(x_pdf, norm=norm)

        params = tuple(param_x_indices.keys())
        with set_values(params, params):
            probs = tf_map(eval_pdf, x_values)
        probs = probs[:, 0]  # removing stack dimension, implicitly in map_fn
        return probs

    def _get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
    ) -> set[ZfitParameter]:
        params = super()._get_params(floating, is_yield, extract_independent)
        params -= set(self._cond)
        return params

    @z.function(wraps="conditional_pdf")
    def _single_hook_integrate(self, limits, norm, x, options):
        from zfit import run

        if not run.get_graph_mode():
            warnings.warn(
                "Using the Conditional PDF in eager mode (no jit) maybe gets stuck.",
                RuntimeWarning,
            )

        param_x_indices = {
            p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()
        }
        x_values = x.value()
        pdf = self.pdfs[0]

        if self._use_vectorized_map:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(
                shape=(1, *x_values.shape[1:-1]), dtype=self.dtype
            )
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)

        @z.function(wraps="vectorized_map")
        def eval_int(values):
            for param, index in param_x_indices.items():
                param.assign(values[..., index])

            return pdf.integrate(limits=limits, norm=norm, options=options)

        integrals = tf_map(eval_int, x_values)
        integrals = integrals[:, 0]  # removing stack dimension, implicitly in map_fn
        return integrals

    @z.function(wraps="conditional_pdf")
    def _single_hook_sample(self, n, limits, x):
        tf.assert_equal(
            n,
            x.nevents,
            message="Different number of n requested than x given for "
            "conditional sampling. Needs to agree",
        )

        param_x_indices = {
            p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()
        }
        x_values = x.value()
        # if self._sample_with_replacement:
        #     x_values = z.random.sample_with_replacement(x_values, axis=0, sample_shape=(n,))
        pdf = self.pdfs[0]

        if self._use_vectorized_map:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, pdf.n_obs), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)

        def eval_sample(values):
            for param, index in param_x_indices.items():
                param.assign(values[..., index])

            return pdf.sample(n=1, limits=limits).value()

        sample_rnd = tf_map(eval_sample, x_values)[..., 0]
        sample = znp.concatenate([sample_rnd, x_values], axis=-1)
        return sample

    def copy(self, **override_parameters) -> BasePDF:
        raise WorkInProgressError(
            "Currently copying not possible. " "Use `set_yield` to set a yield inplace."
        )


# NOT working, logic wrong: the parameter of Gauss is not added to overall variables...
# class ConditionalPDFV1Repr(BasePDFRepr):
#     _implementation = ConditionalPDFV1
#     hs3_type: Literal["ConditionalPDFV1"] = pydantic.Field("ConditionalPDFV1", alias="type")
#
#     pdf: List[Serializer.types.PDFTypeDiscriminated]
#     cond: Dict[Serializer.types.ParamTypeDiscriminated, Union[SpaceRepr, Tuple[str]]]
#     obs: Optional[Union[SpaceRepr, Tuple[str]]] = None
#     extended: Serializer.types.ParamInputTypeDiscriminated = None
#
#     #
#     @pydantic.root_validator(pre=True)
#     def validate_all(cls, values):
#         if cls.orm_mode(values):
#             values = dict(values)
#             for k, v in values['hs3'].original_init.items():
#                 values[k] = v
#             values['pdf'] = [values['pdf']]
#             values['obs'] = values['space']
#         return values
#
#     def _to_orm(self, init):
#         init = dict(init)
#         init['pdf'] = init['pdf'][0]
#         out = super()._to_orm(init)
#         return out

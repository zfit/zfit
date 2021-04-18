#  Copyright (c) 2021 zfit
import functools
from typing import Mapping, Optional, Set

import tensorflow as tf

import zfit.z.numpy as znp

from .. import z
from ..core.interfaces import ZfitIndependentParameter, ZfitPDF, ZfitSpace
from ..core.parameter import set_values
from ..core.space import combine_spaces, convert_to_space
from ..util.exception import WorkInProgressError
from ..util.warnings import warn_experimental_feature
from .functor import BaseFunctor


class ConditionalPDFV1(BaseFunctor):

    @warn_experimental_feature
    def __init__(self,
                 pdf: ZfitPDF,
                 cond: Mapping[ZfitIndependentParameter, ZfitSpace],
                 name: str = "ConditionalPDF",
                 *,
                 use_vectorized_map: bool = False,
                 sample_with_replacement: bool = True
                 ) -> None:
        """EXPERIMENTAL! Implementation of a Conditional PDF, rather slow and for research purpose.

        As an example, a Gaussian is wrapped in order to make 'sigma' conditional.

        .. jupyter-execute::



        Args:
            pdf: PDF that will be wrapped. Convert one or several parameters of *pdf* to a conditional
                parameter, meaning that the parameter *param* in the ``cond`` mapping will now be
                determined by the data in the ``Space``, the value of the ``cond``.
            cond: Mapping of parameter to input data.
            name: |@doc:model.init.name| Name of the PDF for a better visibility.
               Has no programmatical functional purpose. |@docend:model.init.name|
            use_vectorized_map ():
            sample_with_replacement ():
        """
        self._sample_with_replacement = sample_with_replacement
        self._use_vectorized_map = use_vectorized_map
        self._cond, cond_obs = self._check_input_cond(cond)
        obs = pdf.space * cond_obs
        super().__init__(pdfs=pdf, obs=obs, name=name)
        self.set_norm_range(pdf.norm_range)

    def _check_input_cond(self, cond):
        spaces = []
        for param, obs in cond.items():
            if not isinstance(param, ZfitIndependentParameter):
                raise TypeError(f"parameter {param} not a ZfitIndependentParameter")
            spaces.append(convert_to_space(obs))
        return cond, combine_spaces(*spaces)

    @z.function(wraps='conditional_pdf')
    def _pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
        x_values = x.value()

        if self._use_vectorized_map:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, *x_values.shape[1:-1]), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)

        # TODO: reset parameters?

        def eval_pdf(cond_and_data):
            x_pdf = cond_and_data[None, ..., :pdf.n_obs]
            for param, index in param_x_indices.items():
                param.assign(cond_and_data[..., index])
            return pdf.pdf(x_pdf, norm_range=norm_range)

        params = tuple(param_x_indices.keys())
        with set_values(params, params):
            probs = tf_map(eval_pdf, x_values)
        probs = probs[:, 0]  # removing stack dimension, implicitly in map_fn
        return probs

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = super()._get_params(floating, is_yield, extract_independent)
        params -= set(self._cond)
        return params

    @z.function(wraps='conditional_pdf')
    def _single_hook_integrate(self, limits, norm_range, x):

        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
        x_values = x.value()
        pdf = self.pdfs[0]

        if self._use_vectorized_map:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, *x_values.shape[1:-1]), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)

        def eval_int(values):
            for param, index in param_x_indices.items():
                param.assign(values[..., index])

            return pdf.integrate(limits=limits, norm_range=norm_range, x=x)

        integrals = tf_map(eval_int, x_values)
        integrals = integrals[:, 0]  # removing stack dimension, implicitly in map_fn
        return integrals

    @z.function(wraps='conditional_pdf')
    def _single_hook_sample(self, n, limits, x):
        tf.assert_equal(n, x.nevents, message="Different number of n requested than x given for "
                                              "conditional sampling. Needs to agree")

        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
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

    def copy(self, **override_parameters) -> 'BasePDF':
        raise WorkInProgressError("Currently copying not possible. "
                                  "Use `set_yield` to set a yield inplace.")

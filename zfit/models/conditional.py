#  Copyright (c) 2020 zfit
import functools
from typing import Mapping, Optional, Set

import tensorflow as tf

from .functor import BaseFunctor
from ..core.interfaces import ZfitPDF, ZfitSpace, ZfitIndependentParameter
from ..core.space import combine_spaces, convert_to_space


class ConditionalPDFV1(BaseFunctor):

    def __init__(self, pdf: ZfitPDF, cond: Mapping[ZfitIndependentParameter, ZfitSpace], name="ConditionalPDF",
                 *, use_vectorized_map: bool = True):
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

    def _pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        param_x_indices = {p: x.obs.index(x.obs[0]) for p, p_space in self._cond.items()}
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

        probs = tf_map(eval_pdf, x_values)
        probs = probs[:, 0]  # removing stack dimension, implicitly in map_fn
        return probs

    def _get_params(self, floating: Optional[bool] = True, is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set["ZfitParameter"]:
        params = super()._get_params(floating, is_yield, extract_independent)
        params -= set(self._cond)
        return params

    def _single_hook_integrate(self, limits, norm_range, x):
        # return self._hook_partial_integrate(x=x, limits=limits, norm_range=norm_range)

        param_x_indices = {p: x.obs.index(x.obs[0]) for p, p_space in self._cond.items()}
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

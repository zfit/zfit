#  Copyright (c) 2020 zfit
from typing import Mapping

import tensorflow as tf

from .functor import BaseFunctor
from ..core.interfaces import ZfitPDF, ZfitSpace, ZfitIndependentParameter
from ..core.space import combine_spaces, convert_to_space


class ConditionalPDFV1(BaseFunctor):

    def __init__(self, pdf: ZfitPDF, cond: Mapping[ZfitIndependentParameter, ZfitSpace], name="ConditionalPDF"):
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
        call_pdf = pdf.pdf

        # TODO: reset parameters?

        def eval_pdf(cond_and_data):
            x_pdf = cond_and_data[..., :pdf.n_obs]
            for param, index in param_x_indices.items():
                param.assign(cond_and_data[..., index])
            return call_pdf(x_pdf)

        return tf.map_fn(eval_pdf, x.value())
        # return tf.vectorized_map(eval_pdf, x.value())

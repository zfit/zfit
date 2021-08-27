#  Copyright (c) 2021 zfit
from typing import List

import numpy as np
import tensorflow as tf

from .. import z
from ..z import numpy as znp
from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.interfaces import ZfitModel
from ..util.container import convert_to_container
from .basefunctor import FunctorMixin


# class FunctorMixin:
#
#     def __init__(self, models, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self._models = convert_to_container(models)
#
#     @property
#     def models(self) -> List[ZfitModel]:
#         return self._models


class BinnedSumPDFV1(FunctorMixin, BaseBinnedPDFV1):

    def __init__(self, pdfs, obs=None, name="BinnedSumPDF", **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(obs=obs, params={}, name=name, models=pdfs, **kwargs)

        if not all(model.is_extended for model in self.models):
            raise RuntimeError
        self._set_yield(tf.reduce_sum([m.get_yield() for m in self.models]))

    @property
    def _models(self) -> List[ZfitModel]:
        return self.pdfs

    def _unnormalized_pdf(self, x):
        models = self.models
        prob = tf.reduce_sum([model._unnormalized_pdf(x) for model in models], axis=0)

        return prob

    def _ext_pdf(self, x, norm, *, norm_range=None):
        if norm_range is not None:
            norm = norm_range
        prob = tf.reduce_sum([model.ext_pdf(x) for model in self.models], axis=0)
        prob /= np.prod(self.space.binning.width, axis=0)
        return prob

    def _counts(self, x, norm=None):
        if norm not in (None, False, self.norm):
            raise RuntimeError("norm range different than None or False currently not supported.")
        prob = tf.reduce_sum([model.counts(x) for model in self.models], axis=0)
        return prob

    def _rel_counts(self, x, norm=None):
        if norm not in (None, False, self.norm):
            raise RuntimeError("norm range different than None or False currently not supported.")
        counts = self.counts(x, norm)
        return counts / znp.sum(counts)

# class BinnedVariablePDF(FunctorMixin, BaseBinnedPDF):
#
#     def __init__(self, pdf, params):
#         if not isinstance(params, dict):
#             params = {i: p for i, p in enumerate(params)}
#         super().__init__(obs=pdf.space, params=params)
#
#     def _ext_pdf(self, x, norm_range):
#         return z.convert_to_tensor(list(self.params.values()))  # TODO: VectorParam?

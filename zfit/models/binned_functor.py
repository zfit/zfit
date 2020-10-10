#  Copyright (c) 2020 zfit
from typing import List

import tensorflow as tf

from zfit import Data
from zfit.core.binnedpdf import BaseBinnedPDF
from zfit.core.interfaces import ZfitModel
from zfit.util import ztyping


class BinnedSumPDF(BaseBinnedPDF):

    def __init__(self, pdfs, obs=None, name="BinnedSumPDF", **kwargs):
        super().__init__(obs=obs, params={}, name=name, **kwargs)
        self._pdfs = pdfs

        # if not all(model.is_extended for model in self.models):
        #     raise RuntimeError

    def _unnormalized_pdf(self, x):
        models = self.models
        prob = tf.reduce_sum([model._unnormalized_pdf(x) for model in models], axis=0)
        return prob

    def _func_to_integrate(self, x: ztyping.XType) -> tf.Tensor:
        pass

    def _func_to_sample_from(self, x: ztyping.XType) -> Data:
        pass

    @property
    def models(self) -> List[ZfitModel]:
        return self._pdfs

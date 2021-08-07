#  Copyright (c) 2021 zfit
from zfit.core.basepdf import BasePDF
from zfit.models.functor import BaseFunctor


class SplinePDF(BaseFunctor):

    def __init__(self, pdf):
        super().__init__(pdfs=pdf)

    def _unnormalized_pdf(self, x):
        pass

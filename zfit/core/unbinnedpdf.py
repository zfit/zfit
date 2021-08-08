#  Copyright (c) 2021 zfit
import tensorflow_addons as tfa

from zfit.models.functor import BaseFunctor


class SplinePDF(BaseFunctor):

    def __init__(self, pdf, obs):  # TODO: obs should not be needed? Or should it?
        super().__init__(pdfs=pdf, obs=obs)

    def _ext_pdf(self, x):
        pdf = self.pdfs[0]
        density = pdf.ext_pdf(pdf.space, norm=pdf.space)
        centers = pdf.axes.centers[0][None, :, None]  # TODO: only 1 dim now
        y = tfa.image.interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=2

        )
        return y[0, ..., 0]

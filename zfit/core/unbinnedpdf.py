#  Copyright (c) 2021 zfit
import tensorflow_addons as tfa

from zfit.models.functor import BaseFunctor


class SplinePDF(BaseFunctor):

    def __init__(self, pdf, obs=None, extended=None):  # TODO: obs should not be needed? Or should it?
        if pdf.is_extended:
            extended = pdf.get_yield()
        if obs is None:
            obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended)

    def _ext_pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.ext_pdf(x.space, norm=norm_range)  # TODO: order? Give obs, pdf makes order and binning herself?
        centers = pdf.space.binning.centers[0][None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=2

        )
        return probs[0, ..., 0]

    def _pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.pdf(x.space, norm=norm_range)  # TODO: order? Give obs, pdf makes order and binning herself?
        centers = pdf.space.binning.centers[0][None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=2

        )
        return probs[0, ..., 0]

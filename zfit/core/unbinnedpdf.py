#  Copyright (c) 2021 zfit
import pydantic
import tensorflow_addons as tfa

from zfit.models.functor import BaseFunctor
from .space import supports
from ..z import numpy as znp


class SplinePDF(BaseFunctor):

    def __init__(self, pdf, order: int = None, obs=None,
                 extended=None):  # TODO: obs should not be needed? Or should it?
        if pdf.is_extended:
            extended = pdf.get_yield()
        if obs is None:
            obs = pdf.space
        obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended)
        if order is None:
            order = 3
        self._order = order

    @property
    def order(self):
        return self._order

    @supports(norm=True)
    def _ext_pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.ext_pdf(x.space, norm=norm_range)
        density_flat = znp.reshape(density, (-1,))
        centers_list = znp.meshgrid(*pdf.space.binning.centers, indexing='ij')
        centers_list_flat = [znp.reshape(cent, (-1,)) for cent in centers_list]
        centers = znp.stack(centers_list_flat, axis=-1)
        # [None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers[None, ...],
            train_values=density_flat[None, :, None],
            query_points=x.value()[None, ...],
            order=self.order,

        )
        return probs[0, ..., 0]

    @supports(norm=True)
    def _pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.pdf(x.space, norm=norm_range)  # TODO: order? Give obs, pdf makes order and binning herself?
        centers = pdf.space.binning.centers[0][None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=3

        )
        return probs[0, ..., 0]


class TypedSplinePDF(pydantic.BaseModel):
    order: pydantic.conint(ge=0)

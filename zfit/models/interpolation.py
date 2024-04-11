#  Copyright (c) 2024 zfit
from __future__ import annotations

import zfit.z.numpy as znp

from ..core.interfaces import ZfitBinnedPDF
from ..core.space import supports
from ..util import ztyping
from ..util.exception import SpecificFunctionNotImplemented
from ..util.ztyping import ExtendedInputType, NormInputType
from ..z.interpolate_spline import interpolate_spline
from .functor import BaseFunctor


class SplinePDF(BaseFunctor):
    def __init__(
        self,
        pdf: ZfitBinnedPDF,
        order: int | None = None,
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = "SplinePDF",
        label: str | None = None,
    ) -> None:
        """Spline interpolate a binned PDF in order to get a smooth, unbinned PDF.

        Args:
            pdf: Binned PDF that will be interpolated.
            order: Spline interpolation order. Default is 3
            obs: Unbinned observable. If not given, the observable of the pdf is used without the binning.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| The normalization of the PDF. If this is parameter-like, it will be used as the
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        if extended is None:
            extended = pdf.is_extended
        if extended is True:
            extended = pdf.get_yield()
            self._automatically_extended = True
        else:
            self._automatically_extended = False
        if obs is None:
            obs = pdf.space
            obs = obs.with_binning(None)
        if label is None:
            label = f"splined_{pdf.label}"
        super().__init__(pdfs=pdf, obs=obs, extended=extended, norm=norm, name=name, label=label)
        if order is None:
            order = 3
        self._order = order

    @property
    def order(self):
        return self._order

    @supports(norm=True)
    def _ext_pdf(self, x, norm):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        pdf = self.pdfs[0]
        density = pdf.ext_pdf(x.space, norm=norm)
        density_flat = znp.reshape(density, (-1,))
        centers_list = znp.meshgrid(*pdf.space.binning.centers, indexing="ij")
        centers_list_flat = [znp.reshape(cent, (-1,)) for cent in centers_list]
        centers = znp.stack(centers_list_flat, axis=-1)
        # [None, :, None]  # TODO: only 1 dim now
        probs = interpolate_spline(
            train_points=centers[None, ...],
            train_values=density_flat[None, :, None],
            query_points=x.value()[None, ...],
            order=self.order,
        )
        return probs[0, ..., 0]

    @supports(norm=True)
    def _pdf(self, x, norm):
        pdf = self.pdfs[0]
        density = pdf.pdf(x.space, norm=norm)  # TODO: order? Give obs, pdf makes order and binning herself?
        centers = pdf.space.binning.centers[0][None, :, None]  # TODO: only 1 dim now
        probs = interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=3,
        )
        return probs[0, ..., 0]

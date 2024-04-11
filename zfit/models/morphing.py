#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable, Mapping

import tensorflow as tf
from uhi.typing.plottable import PlottableHistogram

import zfit.z.numpy as znp
from zfit import z
from zfit.core.binnedpdf import BaseBinnedPDFV1

from ..core import parameter
from ..core.interfaces import ZfitBinnedPDF
from ..util import ztyping
from ..util.exception import SpecificFunctionNotImplemented
from ..z.interpolate_spline import interpolate_spline


@z.function(wraps="tensor", keepalive=True)
def spline_interpolator(alpha, alphas, densities):
    alphas = alphas[None, :, None]
    shape = tf.shape(densities[0])
    densities_flat = [znp.reshape(density, [-1]) for density in densities]
    densities_flat = znp.stack(densities_flat, axis=0)
    alpha_shaped = znp.reshape(alpha, [1, -1, 1])

    y_flat = interpolate_spline(
        train_points=alphas,
        train_values=densities_flat[None, ...],
        query_points=alpha_shaped,
        order=2,
    )
    y_flat = y_flat[0, 0]
    return znp.reshape(y_flat, shape)


class SplineMorphingPDF(BaseBinnedPDFV1):
    _morphing_interpolator = staticmethod(spline_interpolator)

    def __init__(
        self,
        alpha: ztyping.ParamTypeInput,
        hists: (Mapping[float | int, Iterable[ZfitBinnedPDF]] | list[ZfitBinnedPDF] | tuple[ZfitBinnedPDF]),
        *,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
        name: str | None = "SplineMorphingPDF",
        label: str | None = None,
    ):
        """Morphing a set of histograms with a spline interpolation.

        Args:
            alpha: Parameter for the spline interpolation.
            hists: A mapping of alpha values to histograms. This allows for arbitrary interpolation points.
                If a list or tuple of exactly three PDFs is given, this corresponds to the histograms at
                alhpa equal to -1, 0 and 1 respectively.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        if isinstance(hists, (list, tuple)):
            if len(hists) != 3:
                msg = (
                    "If hists is a list, it is assumed to correspond to an alpha of -1, 0 and 1."
                    f" hists is {hists} and has length {len(hists)}."
                )
                raise ValueError(msg)
            hists = {float(i - 1): hist for i, hist in enumerate(hists)}  # mapping to -1, 0, 1

        for a, hist in hists.items():
            if isinstance(hist, PlottableHistogram):
                from zfit.models.histogram import HistogramPDF

                hist = HistogramPDF(hist)
            if isinstance(hist, ZfitBinnedPDF):
                hists[a] = hist
            else:
                msg = f"hist {hist} is not a ZfitBinnedPDF or a UHI histogram."
                raise TypeError(msg)

        self.hists = hists
        self.alpha = alpha
        obs = next(iter(hists.values())).space
        all_extended = all(hist.is_extended for hist in hists.values())
        if extended is None:  # TODO: yields?
            extended = all_extended
        self._automatically_extended = None
        if extended is True:  # create the yield automatically
            self._automatically_extended = True
            if not all_extended:
                msg = "If extended is True, all PDFs must be extended to create the yield automatically."
                raise ValueError(msg)
            alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)

            def interpolated_yield(params):
                alpha = params["alpha"]
                densities = tuple(
                    params[f"{i}"] for i in range(len(params) - 1)
                )  # params has n hist entries + 1 alpha entry
                return spline_interpolator(alpha=alpha, alphas=alphas, densities=densities)

            number = parameter.get_auto_number()
            yields = {f"{i}": hist.get_yield() for i, hist in enumerate(hists.values())}
            yields["alpha"] = alpha
            new_yield = parameter.ComposedParameter(
                f"AUTOGEN_{number}_interpolated_yield",
                interpolated_yield,
                params=yields,
            )
            extended = new_yield
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(obs=obs, extended=extended, norm=norm, params={"alpha": alpha}, name=name, label=label)

    def _counts(self, x, norm):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        densities = [hist.counts(x, norm=norm) for hist in self.hists.values()]
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)
        alpha = self.params["alpha"].value()
        return self._morphing_interpolator(alpha, alphas, densities)

    def _rel_counts(self, x, norm):
        densities = [hist.rel_counts(x, norm=norm) for hist in self.hists.values()]
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)
        alpha = self.params["alpha"].value()
        return self._morphing_interpolator(alpha, alphas, densities)

    def _ext_pdf(self, x, norm):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        densities = [hist.ext_pdf(x, norm=norm) for hist in self.hists.values()]
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)
        alpha = self.params["alpha"].value()
        return self._morphing_interpolator(alpha, alphas, densities)

    def _pdf(self, x, norm):
        densities = [hist.pdf(x, norm=norm) for hist in self.hists.values()]
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)
        alpha = self.params["alpha"].value()
        return self._morphing_interpolator(alpha, alphas, densities)

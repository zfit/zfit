#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

from uhi.typing.plottable import PlottableHistogram

from .. import z
from ..core.binnedpdf import BaseBinnedPDF
from ..core.interfaces import ZfitPDF
from ..core.space import supports
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_norm_range
from ..util.exception import NormNotImplemented
from ..util.ztyping import BinnedDataInputType
from ..z import numpy as znp
from .basefunctor import FunctorMixin, _preprocess_init_sum


def preprocess_pdf_or_hist(models: Union[ZfitPDF, Iterable[ZfitPDF], BinnedDataInputType]):
    models = convert_to_container(models)
    from zfit.models.histogram import HistogramPDF

    return [HistogramPDF(model) if isinstance(model, PlottableHistogram) else model for model in models]


class BaseBinnedFunctorPDF(FunctorMixin, BaseBinnedPDF):
    """Base class for binned functors."""

    def __init__(self, models, obs, **kwargs):
        models = preprocess_pdf_or_hist(models)
        super().__init__(models, obs, **kwargs)
        self.pdfs = self.models


class BinnedSumPDF(BaseBinnedFunctorPDF):
    def __init__(
        self,
        pdfs: ztyping.BinnedHistPDFInputType,
        fracs: ztyping.ParamTypeInput | None = None,
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ztyping.ExtendedInputType = None,
        name: str = "BinnedSumPDF",
        label: str | None = None,
    ):
        """Sum of binned PDFs.

        Args:
            pdfs: Binned PDFs to sum.
            fracs: Fractions of the PDFs. If not given, they are created as parameters.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            name: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        self._fracs = None

        pdfs = preprocess_pdf_or_hist(pdfs)
        self.pdfs = pdfs

        (
            all_extended,
            fracs_cleaned,
            param_fracs,
            params,
            sum_yields,
            frac_param_created,
        ) = _preprocess_init_sum(fracs, obs, pdfs)
        del frac_param_created  # currently actually not used

        self._fracs = param_fracs
        self._original_fracs = fracs_cleaned

        if extended is None:
            extended = all_extended
        if extended is True:
            extended = sum_yields if all_extended else None

        super().__init__(models=pdfs, obs=obs, params=params, name=name, extended=extended, label=label)

    @supports(norm=True)
    def _pdf(self, x, norm):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        pdfs = self.pdfs
        fracs = self.params.values()
        probs = []
        for pdf, frac in zip(pdfs, fracs):
            probs.append(pdf.pdf(x) * frac)
        prob = znp.sum(probs, axis=0)
        return z.convert_to_tensor(prob)

    @deprecated_norm_range
    def _ext_pdf(self, x, norm):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        prob = znp.sum([model.ext_pdf(x) for model in self.models], axis=0)
        return z.convert_to_tensor(prob)

    def _counts(self, x, norm=None):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        return znp.sum([model.counts(x) for model in self.models], axis=0)

    def _rel_counts(self, x, norm=None):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        fracs = self.params.values()
        return znp.sum(
            [model.rel_counts(x) * frac for model, frac in zip(self.models, fracs)],
            axis=0,
        )

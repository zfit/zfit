#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Iterable

from .basefunctor import FunctorMixin, _preprocess_init_sum
from .. import z
from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.interfaces import ZfitPDF
from ..core.space import supports
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_norm_range
from ..util.exception import NormNotImplemented
from ..z import numpy as znp


class BaseBinnedFunctorPDF(FunctorMixin, BaseBinnedPDFV1):
    """Base class for binned functors."""

    def __init__(self, models, obs, **kwargs):
        super().__init__(models, obs, **kwargs)
        self.pdfs = self.models


class BinnedSumPDF(BaseBinnedFunctorPDF):
    def __init__(
        self,
        pdfs: Iterable[ZfitPDF],
        fracs: ztyping.ParamTypeInput | None = None,
        obs: ztyping.ObsTypeInput = None,
        name: str = "BinnedSumPDF",
    ):
        self._fracs = None

        pdfs = convert_to_container(pdfs)
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

        extended = sum_yields if all_extended else None
        super().__init__(
            models=pdfs, obs=obs, params=params, name=name, extended=extended
        )

    # def _unnormalized_pdf(self, x):
    #     models = self.models
    #     prob = tf.reduce_sum([model._unnormalized_pdf(x) for model in models], axis=0)
    #     return prob

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
    def _ext_pdf(self, x, norm, *, norm_range=None):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        prob = znp.sum([model.ext_pdf(x) for model in self.models], axis=0)
        return z.convert_to_tensor(prob)

    def _counts(self, x, norm=None):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        prob = znp.sum([model.counts(x) for model in self.models], axis=0)
        return prob

    def _rel_counts(self, x, norm=None):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise NormNotImplemented
        fracs = self.params.values()
        prob = znp.sum(
            [model.rel_counts(x) * frac for model, frac in zip(self.models, fracs)],
            axis=0,
        )
        return prob

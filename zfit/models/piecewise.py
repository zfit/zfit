#  Copyright (c) 2024 zfit
from typing import Literal

import pydantic
import tensorflow as tf
from .basefunctor import FunctorPDFRepr
from .functor import BaseFunctor
from .. import supports, z
from ..core.serialmixin import SerializableMixin
from ..serialization import SpaceRepr
from ..util.container import convert_to_container
import zfit.z.numpy as znp
from ..util.exception import SpecificFunctionNotImplemented


class TruncatedPDF(BaseFunctor, SerializableMixin):
    def __init__(
        self, pdf, limits, obs=None, norms=None, extended=None, name="PiecewisePDF"
    ):
        """Truncated PDF in one or multiple ranges.

        Args:
            pdf: The PDF to be truncated.
            limits: The limits to truncate the PDF. Can be a single or multiple limits.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
               Can be a single or multiple norms.
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
        """
        self._limits = convert_to_container(limits)
        self._norms = convert_to_container(
            norms
        )  # TODO: check if space etc, get min/max of limits
        super().__init__(obs=obs, name=name, extended=extended, norm=None, pdfs=pdf)

    @property
    def limits(self):
        return self._limits

    @property
    def norms(self):
        return self._norms

    def _unnormalized_pdf(self, x):
        prob = self.pdfs[0].pdf(x, norm=False)
        inside_arrays = [limit.inside(x) for limit in self._limits]
        prob *= znp.asarray(znp.any(inside_arrays, axis=0), dtype=znp.float64)
        return prob

    @supports(norm=True)
    def _normalization(self, norm, options):
        norms = convert_to_container(norm)  # todo: what's the best way here?
        normterms = [self.pdfs[0].normalization(norm) for norm in norms]
        return znp.sum(normterms, axis=0)

    @supports()
    def _integrate(self, limits, norm, options=None):
        if (
            limits != self.space
        ):  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        integrals = [
            self.pdfs[0].integrate(limits=limit, norm=False) for limit in limits
        ]
        integral = znp.sum(integrals, axis=0)
        return integral

    # TODO: we could make sampling more efficient by only sampling the relevant ranges, however, that would
    # mean we need to check if the limits of the pdf are within the limits given
    @supports()
    def _sample(self, n, limits):
        pdf = self.pdfs[0]
        if (
            limits != self.space
        ):  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        integrals = znp.concatenate(
            [pdf.integrate(limits=limit, norm=False) for limit in limits]
        )
        fracs = integrals / znp.sum(integrals, axis=0)  # norm
        fracs.set_shape([len(limits)])
        counts = tf.unstack(z.random.counts_multinomial(n, probs=fracs), axis=0)
        samples = [
            self.pdfs[0].sample(count, limits=limit).value()
            for count, limit in zip(counts, limits)
        ]
        return znp.concatenate(samples, axis=0)


class TruncatedPDFRepr(FunctorPDFRepr):
    _implementation = TruncatedPDF
    hs3_type: Literal["TruncatedPDF"] = pydantic.Field("TruncatedPDF", alias="type")
    limits: list[SpaceRepr]
    norms: list[SpaceRepr]

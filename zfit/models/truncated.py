#  Copyright (c) 2024 zfit
from typing import List, Literal, Union

import numpy as np
import pydantic
import tensorflow as tf

import zfit.z.numpy as znp

from .. import z
from ..core.interfaces import ZfitSpace
from ..core.serialmixin import SerializableMixin
from ..core.space import supports
from ..serialization import Serializer, SpaceRepr  # noqa: F401
from ..util.container import convert_to_container
from ..util.exception import SpecificFunctionNotImplemented
from .basefunctor import FunctorPDFRepr
from .functor import BaseFunctor


def check_limits(limits: Union[ZfitSpace, List[ZfitSpace]]):
    """Check if the limits are valid Spaces and return an iterable."""
    limits = convert_to_container(limits, container=tuple)
    if limits is not None:
        notspace = [limit for limit in limits if not isinstance(limit, ZfitSpace)]
        if notspace:
            msg = f"limits {notspace} are not of type ZfitSpace."
            raise TypeError(msg)
    return limits


class TruncatedPDF(BaseFunctor, SerializableMixin):
    def __init__(self, pdf, limits, obs=None, norms=None, extended=None, name="PiecewisePDF"):
        """Truncated PDF in one or multiple ranges.

        The PDF is truncated to the given limits, i.e. the PDF is only evaluated within the given limits
        and outside the limits, the PDF is zero. The limits can be given as a single or as a list of limits.
        This solves two problems: first, the PDF is only evaluated where it is defined and zero outside,
        which enables PDFs that are not defined outside the limits to be used (Poisson, LogNormal, Chi2,...).
        Second, the list of limits can be used to effectively create a multilimit PDF, that means, a PDF that
        is defined in multiple ranges and zero outside these ranges, also in between.

        Historically, this was implemented in zfit through MultipleLimits, which is now deprecated and replaced
        by this class.

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
        original_init = {"extended": extended, "obs": obs}

        self._limits = check_limits(limits)
        self._norms = check_limits(norms)  # TODO: check if space etc, get min/max of limits?
        if obs is None:
            obs = pdf.space
        if extended is True and pdf.is_extended:
            msg = "Cannot automatically take the value, would need to integrate and get correct fraction."
            raise ValueError(msg)
            # extended = pdf.get_yield()  # TODO: that's probably not quite right per se?
        super().__init__(obs=obs, name=name, extended=extended, norm=None, pdfs=pdf)
        self.hs3.original_init.update(original_init)

    @property
    def limits(self):
        return self._limits

    @property
    def norms(self):
        return self._norms

    def _unnormalized_pdf(self, x):
        # the implementation only feeds the pdf with data that is inside the limits that we want to evaluate
        # this maybe has a speedup (although limited as we're using dynamic shapes -> needs to change for JAX)
        # but most importantly avoids any issue if we take the gradient of a pdf that is not defined outside
        # the limits, because this can yield NaNs using a naive multiplication with a mask
        from zfit import Data

        xarray = znp.asarray(x.value())
        inside_arrays = [limit.inside(x) for limit in self._limits]
        inside = znp.any(inside_arrays, axis=0)
        indices = znp.transpose(znp.where(inside))  # is a list? but transpose gives perfect shape for indices
        data = Data.from_tensor(tensor=xarray[inside], obs=self.obs)
        prob = self.pdfs[0].pdf(data, norm=False)
        return tf.scatter_nd(indices, prob, tf.shape(xarray, out_type=np.int64)[:1])  # only nevents

    @supports(norm=True)
    def _normalization(self, norm, options):
        if (norms := self._norms) is None:
            norms = [norm]
        elif norm != self.space:
            msg = f"Cannot normalize to a different space than the one given, the norms {norms}."
            raise RuntimeError(msg)

        normterms = [self.pdfs[0].normalization(norm) for norm in norms]
        return znp.sum(normterms, axis=0)

    @supports()
    def _integrate(self, limits, norm, options=None):
        if limits != self.space:  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        integrals = [self.pdfs[0].integrate(limits=limit, norm=False) for limit in limits]
        return znp.sum(integrals, axis=0)

    # TODO: we could make sampling more efficient by only sampling the relevant ranges, however, that would
    # mean we need to check if the limits of the pdf are within the limits given
    @supports()
    def _sample(self, n, limits):
        pdf = self.pdfs[0]
        if limits != self.space:  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        integrals = znp.concatenate([pdf.integrate(limits=limit, norm=False) for limit in limits])
        fracs = integrals / znp.sum(integrals, axis=0)  # norm
        fracs.set_shape([len(limits)])
        counts = tf.unstack(z.random.counts_multinomial(n, probs=fracs), axis=0)
        samples = [self.pdfs[0].sample(count, limits=limit).value() for count, limit in zip(counts, limits)]
        return znp.concatenate(samples, axis=0)


class TruncatedPDFRepr(FunctorPDFRepr):
    _implementation = TruncatedPDF
    hs3_type: Literal["TruncatedPDF"] = pydantic.Field("TruncatedPDF", alias="type")
    limits: List[SpaceRepr]

    def _to_orm(self, init):
        init["pdf"] = init.pop("pdfs")[0]
        return super()._to_orm(init)

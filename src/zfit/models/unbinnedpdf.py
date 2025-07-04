#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

import pydantic.v1 as pydantic
import tensorflow as tf

from zfit import z
from zfit._interfaces import ZfitSpace
from zfit.z import numpy as znp

from ..core.binning import unbinned_to_binindex
from ..core.space import supports
from ..util.ztyping import ExtendedInputType, NormInputType
from .functor import BaseFunctor

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class UnbinnedFromBinnedPDF(BaseFunctor):
    def __init__(
        self,
        pdf,
        obs=None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = "UnbinnedFromBinnedPDF",
        label: str | None = None,
    ):
        """Create a unbinned pdf from a binned pdf.

        Args:
            pdf: Binned PDF that will be used as a step function.
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
        if extended is None and extended is not False and pdf.is_extended:
            extended = pdf.get_yield()
        if obs is None:
            obs = pdf.space
            obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended, norm=norm, name=name, label=label)
        if label is None:
            label = f"unbinned_{pdf.label}"
        self._binned_space = self.pdfs[0].space.with_obs(self.space)
        self._binned_norm = self.pdfs[0].norm.with_obs(self.space)

    @supports(norm="norm", multiple_limits=True)
    def _pdf(self, x, norm):
        binned_space = self.pdfs[0].space
        binindices = unbinned_to_binindex(x, binned_space, flow=True)
        pdf = self.pdfs[0]

        binned_norm = norm if norm is False else self._binned_norm
        values = pdf.pdf(binned_space, norm=binned_norm)

        # because we have the flow, so we need to make it here with pads
        padded_values = znp.pad(values, znp.ones((z._get_ndims(values), 2)), mode="constant")  # for overflow
        return tf.gather_nd(padded_values, indices=binindices)

    @supports(norm="norm", multiple_limits=True)
    def _ext_pdf(self, x, norm):
        binned_space = self.pdfs[0].space
        binindices = unbinned_to_binindex(x, binned_space, flow=True)

        pdf = self.pdfs[0]

        binned_norm = norm if norm is False else self._binned_norm
        values = pdf.ext_pdf(binned_space, norm=binned_norm)
        ndim = z._get_ndims(values)

        # because we have the flow, so we need to make it here with pads
        padded_values = znp.pad(values, znp.ones((ndim, 2)), mode="constant")  # for overflow
        return tf.gather_nd(padded_values, indices=binindices)

    @supports(norm="norm", multiple_limits=True)
    def _integrate(self, limits, norm, options=None):
        binned_norm = norm if norm is False else self._binned_norm
        return self.pdfs[0].integrate(limits, norm=binned_norm, options=options)

    @supports(norm=True, multiple_limits=True)
    def _ext_integrate(self, limits, norm, options):
        binned_norm = norm if norm is False else self._binned_norm
        return self.pdfs[0].ext_integrate(limits, norm=binned_norm, options=options)

    @supports(norm=True, multiple_limits=True)
    def _sample(self, n, limits: ZfitSpace, *, prng=None):  # noqa: ARG002
        if prng is None:
            prng = z.random.get_prng()

        pdf = self.pdfs[0]
        # TODO: use real limits, currently not supported in binned sample
        sample = pdf.sample(n=n)

        edges = sample.space.binning.edges
        ndim = len(edges)
        edges = [znp.array(edge) for edge in edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing="ij")
        uppers_meshed = znp.meshgrid(*uppers, indexing="ij")
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)

        counts_flat = znp.reshape(sample.values(), (-1,))
        counts_flat = znp.asarray(counts_flat, znp.int32)  # TODO: what if we have fractions?
        lower_flat_repeated = tf.repeat(lower_flat, counts_flat, axis=0)
        upper_flat_repeated = tf.repeat(upper_flat, counts_flat, axis=0)
        return prng.uniform(
            (znp.sum(counts_flat), ndim),
            minval=lower_flat_repeated,
            maxval=upper_flat_repeated,
            dtype=self.dtype,
        )


class TypedSplinePDF(pydantic.BaseModel):
    order: pydantic.conint(ge=0)

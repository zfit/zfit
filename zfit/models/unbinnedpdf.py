#  Copyright (c) 2023 zfit
import pydantic
import tensorflow as tf

from zfit import z
from zfit.core.binning import unbinned_to_binindex
from zfit.core.interfaces import ZfitSpace
from zfit.core.space import supports
from zfit.models.functor import BaseFunctor
from zfit.z import numpy as znp


class UnbinnedFromBinnedPDF(BaseFunctor):
    def __init__(self, pdf, obs=None):
        """Create a unbinned pdf from a binned pdf.

        Args:
            pdf:
            obs:
        """
        if pdf.is_extended:
            extended = pdf.get_yield()
        else:
            extended = None
        if obs is None:
            obs = pdf.space
            obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended)
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
        padded_values = znp.pad(
            values, znp.ones((z._get_ndims(values), 2)), mode="constant"
        )  # for overflow
        ordered_values = tf.gather_nd(padded_values, indices=binindices)
        return ordered_values

    @supports(norm="norm", multiple_limits=True)
    def _ext_pdf(self, x, norm):
        binned_space = self.pdfs[0].space
        binindices = unbinned_to_binindex(x, binned_space, flow=True)

        pdf = self.pdfs[0]

        binned_norm = norm if norm is False else self._binned_norm
        values = pdf.ext_pdf(binned_space, norm=binned_norm)
        ndim = z._get_ndims(values)

        # because we have the flow, so we need to make it here with pads
        padded_values = znp.pad(
            values, znp.ones((ndim, 2)), mode="constant"
        )  # for overflow
        ordered_values = tf.gather_nd(padded_values, indices=binindices)
        return ordered_values

    @supports(norm="norm", multiple_limits=True)
    def _integrate(self, limits, norm, options=None):
        binned_norm = norm if norm is False else self._binned_norm
        return self.pdfs[0].integrate(limits, norm=binned_norm, options=options)

    @supports(norm=True, multiple_limits=True)
    def _ext_integrate(self, limits, norm, options):
        binned_norm = norm if norm is False else self._binned_norm
        return self.pdfs[0].ext_integrate(limits, norm=binned_norm, options=options)

    @supports(norm=True, multiple_limits=True)
    def _sample(self, n, limits: ZfitSpace, *, prng=None):
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
        lowers_meshed_flat = [
            znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed
        ]
        uppers_meshed_flat = [
            znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed
        ]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)

        counts_flat = znp.reshape(sample.values(), (-1,))
        counts_flat = tf.cast(
            counts_flat, znp.int32
        )  # TODO: what if we have fractions?
        lower_flat_repeated = tf.repeat(lower_flat, counts_flat, axis=0)
        upper_flat_repeated = tf.repeat(upper_flat, counts_flat, axis=0)
        sample_unbinned = prng.uniform(
            (znp.sum(counts_flat), ndim),
            minval=lower_flat_repeated,
            maxval=upper_flat_repeated,
            dtype=self.dtype,
        )
        return sample_unbinned


class TypedSplinePDF(pydantic.BaseModel):
    order: pydantic.conint(ge=0)

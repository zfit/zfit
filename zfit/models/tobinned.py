#  Copyright (c) 2023 zfit
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit import z
from .binned_functor import BaseBinnedFunctorPDF
from ..core.interfaces import ZfitPDF, ZfitSpace
from ..util import ztyping
from ..util.warnings import warn_advanced_feature


class BinnedFromUnbinnedPDF(BaseBinnedFunctorPDF):
    def __init__(
        self,
        pdf: ZfitPDF,
        space: ZfitSpace,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
    ) -> None:
        """Create a binned pdf from an unbinned pdf binning in *space*.

        Args:
            pdf: The unbinned pdf to be binned.
            space: |@doc:pdf.init.obs| Observables of the
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
        """
        if pdf.is_extended:
            if extended is not None:
                warn_advanced_feature(
                    f"PDF {pdf} is already extended, but extended also given {extended}. Will"
                    f" use the given yield.",
                    identifier="extend_wrapped_extended",
                )
            else:
                extended = pdf.get_yield()
        if not isinstance(space, ZfitSpace):
            try:
                space = pdf.space.with_binning(space)
            except Exception as error:
                raise ValueError(
                    f"Could not create space {space} from pdf {pdf} with binning {space}"
                ) from error
        super().__init__(
            obs=space,
            extended=extended,
            norm=norm,
            models=pdf,
            params={},
            name="BinnedFromUnbinnedPDF",
        )
        self.pdfs = self.models

    # def _get_params(self, floating: bool | None = True, is_yield: bool | None = None,
    #                 extract_independent: bool | None = True) -> set[ZfitParameter]:
    #     params = super()._get_params(floating=floating, is_yield=is_yield, extract_independent=extract_independent)
    #     daughter_params = self.pdfs[0].get_params(floating=floating, is_yield=is_yield,
    #                                               extract_independent=extract_independent)
    #     return daughter_params | params

    @z.function
    def _rel_counts(self, x, norm):
        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing="ij")
        uppers_meshed = znp.meshgrid(*uppers, indexing="ij")
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [
            znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed
        ]
        uppers_meshed_flat = [
            znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed
        ]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {"type": "bins"}

        @z.function
        def integrate_one(limits):
            l, u = tf.unstack(limits)
            limits_space = zfit.Space(obs=self.obs, limits=[l, u])
            return pdf.integrate(limits_space, norm=False, options=options)

        limits = znp.stack([lower_flat, upper_flat], axis=1)
        from zfit import run

        try:
            if run.executing_eagerly():
                raise TypeError("Just stearing the eager execution")
            values = tf.vectorized_map(integrate_one, limits)[:, 0]
        except (ValueError, TypeError):
            with run.aquire_cpu(-1) as cpus:
                values = tf.map_fn(integrate_one, limits, parallel_iterations=len(cpus))
        values = znp.reshape(values, shape)
        if norm:
            values /= pdf.normalization(norm)
        return values

    @z.function(wraps="model_binned")
    def _counts(self, x, norm):
        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing="ij")
        uppers_meshed = znp.meshgrid(*uppers, indexing="ij")
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [
            znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed
        ]
        uppers_meshed_flat = [
            znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed
        ]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {"type": "bins"}

        if pdf.is_extended:

            @z.function
            def integrate_one(limits):
                l, u = tf.unstack(limits)
                limits_space = zfit.Space(obs=self.obs, limits=[l, u])
                return pdf.ext_integrate(limits_space, norm=False, options=options)

            missing_yield = False
        else:

            @z.function
            def integrate_one(limits):
                l, u = tf.unstack(limits)
                limits_space = zfit.Space(obs=self.obs, limits=[l, u])
                return pdf.integrate(limits_space, norm=False, options=options)

            missing_yield = True

        limits = znp.stack([lower_flat, upper_flat], axis=1)
        from zfit import run

        try:
            if run.executing_eagerly():
                raise TypeError("Just stearing the eager execution")
            values = tf.vectorized_map(integrate_one, limits)[:, 0]
        except (ValueError, TypeError):
            with run.aquire_cpu(-1) as cpus:
                values = tf.map_fn(integrate_one, limits, parallel_iterations=len(cpus))
        values = znp.reshape(values, shape)
        if missing_yield:
            values *= self.get_yield()
        if norm:
            values /= pdf.normalization(norm)
        return values

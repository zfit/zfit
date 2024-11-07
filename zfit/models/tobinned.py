#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zfit

import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit import z

from ..core.interfaces import ZfitPDF, ZfitSpace
from ..core.space import supports
from ..util import ztyping
from ..util.warnings import warn_advanced_feature
from .binned_functor import BaseBinnedFunctorPDF


class MapNotVectorized(Exception):
    pass


class BinnedFromUnbinnedPDF(BaseBinnedFunctorPDF):
    def __init__(
        self,
        pdf: ZfitPDF,
        space: ZfitSpace,
        *,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        """Create a binned pdf from an unbinned pdf binning in *space*.

        Args:
            pdf: The unbinned pdf to be binned.
            space: |@doc:pdf.init.obs| Observables of the
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
        self._use_vectorized_map = None
        if pdf.is_extended:
            if extended is not None:
                warn_advanced_feature(
                    f"PDF {pdf} is already extended, but extended also given {extended}. Will" f" use the given yield.",
                    identifier="extend_wrapped_extended",
                )
            else:
                extended = pdf.get_yield()
        if not isinstance(space, ZfitSpace):
            try:
                space = pdf.space.with_binning(space)
            except Exception as error:
                msg = f"Could not create space {space} from pdf {pdf} with binning {space}"
                raise ValueError(msg) from error
        if label is None:
            label = f"Binned_{pdf.name}"
        super().__init__(
            obs=space,
            extended=extended,
            norm=norm,
            models=pdf,
            params={},
            name=name,
            label=label,
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
        del x  # not used, we just return the full histogram
        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing="ij")
        uppers_meshed = znp.meshgrid(*uppers, indexing="ij")
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {"type": "bins"}

        @z.function
        def integrate_one(limits, *, obs=self.obs, pdf=pdf, options=options):
            low, up = tf.unstack(limits)
            limits_space = zfit.Space(obs=obs, limits=[low, up])
            return pdf.integrate(limits_space, norm=False, options=options)

        limits = znp.stack([lower_flat, upper_flat], axis=1)
        from zfit import run

        vectorized = self._use_vectorized_map or (self._use_vectorized_map is not False and pdf.has_analytic_integral)
        try:
            if run.get_graph_mode() is False:  #  we cannot use the vectorized version, as it jit compiles
                # also, the map_fn is slower...
                msg = "Just stearing the eager execution"
                raise MapNotVectorized(msg)
            if vectorized:  # noqa: SIM108
                values = tf.vectorized_map(integrate_one, limits)[:, 0]
            else:
                values = tf.map_fn(integrate_one, limits)  # this works
        except (ValueError, MapNotVectorized):
            values = znp.asarray(tuple(map(integrate_one, limits)))
        values = znp.reshape(values, shape)
        if norm:
            values /= pdf.normalization(norm)
        return values

    @z.function(wraps="model_binned")
    @supports(norm="space")
    def _counts(self, x, norm):
        del x  # not used, we just return the full histogram

        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing="ij")
        uppers_meshed = znp.meshgrid(*uppers, indexing="ij")
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {"type": "bins"}

        if pdf.is_extended:

            @z.function
            def integrate_one(limits):
                low, up = tf.unstack(limits)
                limits_space = zfit.Space(obs=self.obs, limits=[low, up])
                return pdf.ext_integrate(limits_space, norm=False, options=options)

            missing_yield = False
        else:

            @z.function
            def integrate_one(limits):
                low, up = tf.unstack(limits)
                limits_space = zfit.Space(obs=self.obs, limits=[low, up])
                return pdf.integrate(limits_space, norm=False, options=options)

            missing_yield = True

        limits = znp.stack([lower_flat, upper_flat], axis=1)
        from zfit import run

        vectorized = self._use_vectorized_map or (self._use_vectorized_map is not False and pdf.has_analytic_integral)
        try:
            if run.get_graph_mode() is False:  #  we cannot use the vectorized version, as it jit compiles
                # also, the map_fn is slower...
                msg = "Just stearing the eager execution"
                raise MapNotVectorized(msg)
            if vectorized:  # noqa: SIM108
                values = tf.vectorized_map(integrate_one, limits)[:, 0]
            else:
                values = tf.map_fn(integrate_one, limits)  # this works
        except (ValueError, MapNotVectorized):
            values = znp.asarray(tuple(map(integrate_one, limits)))
        values = znp.reshape(values, shape)
        if missing_yield:
            values *= self.get_yield()
        if norm:
            values /= pdf.normalization(norm)
        return values

    def __str__(self):
        return f"<Binned {self.pdfs[0]} binning={self.space.binning}>"

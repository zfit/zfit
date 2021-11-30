#  Copyright (c) 2021 zfit
from __future__ import annotations

import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit import z
from .binned_functor import BaseBinnedFunctorPDF
from ..core.interfaces import ZfitParameter
from ..util.warnings import warn_advanced_feature


class BinnedFromUnbinnedPDF(BaseBinnedFunctorPDF):

    def __init__(self, pdf, space, extended=None, norm=None):
        if pdf.is_extended:
            if extended is not None:
                warn_advanced_feature(f"PDF {pdf} is already extended, but extended also given {extended}. Will"
                                      f" use the given yield.", identifier="extend_wrapped_extended")
            else:
                extended = pdf.get_yield()
        super().__init__(obs=space, extended=extended, norm=norm, params={}, name="BinnedFromUnbinnedPDF")
        self.pdfs = [pdf]

    def _get_params(self, floating: bool | None = True, is_yield: bool | None = None,
                    extract_independent: bool | None = True) -> set[ZfitParameter]:
        params = super()._get_params(floating=floating, is_yield=is_yield, extract_independent=extract_independent)
        daughter_params = self.pdfs[0].get_params(floating=floating, is_yield=is_yield,
                                                  extract_independent=extract_independent)
        return daughter_params | params

    @z.function
    def _rel_counts(self, x, norm):
        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing='ij')
        uppers_meshed = znp.meshgrid(*uppers, indexing='ij')
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {'type': 'bins'}

        @z.function
        def integrate_one(limits):
            l, u = tf.unstack(limits)
            limits_space = zfit.Space(obs=self.obs, limits=[l, u])
            return pdf.integrate(limits_space, norm=False, options=options)

        limits = znp.stack([lower_flat, upper_flat], axis=1)
        # values = tf.map_fn(integrate_one, limits)
        values = tf.vectorized_map(integrate_one, limits)[:, 0]
        values = znp.reshape(values, shape)
        if norm:
            values /= pdf.normalization(norm)
        return values

    @z.function
    def _counts(self, x, norm):
        pdf = self.pdfs[0]
        edges = [znp.array(edge) for edge in self.axes.edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing='ij')
        uppers_meshed = znp.meshgrid(*uppers, indexing='ij')
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)
        options = {'type': 'bins'}

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
        # values = tf.map_fn(integrate_one, limits)  # HACK
        # print('HACK INPLACE binnedpdf')
        # values = znp.array([integrate_one(limit) for limit in limits])  # HACK
        try:
            values = tf.vectorized_map(integrate_one, limits)[:, 0]
        except ValueError:
            values = tf.map_fn(integrate_one, limits)
        values = znp.reshape(values, shape)
        if missing_yield:
            values *= self.get_yield()
        if norm:
            values /= pdf.normalization(norm)
        return values

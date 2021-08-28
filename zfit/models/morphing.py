#  Copyright (c) 2021 zfit
import tensorflow as tf
import tensorflow_addons as tfa

import zfit.z.numpy as znp
from zfit.core.binnedpdf import BaseBinnedPDFV1


class SplineMorphing(BaseBinnedPDFV1):
    def __init__(self, alpha, hists, extended=None, norm=None):

        if isinstance(hists, list):
            if len(hists) != 3:
                raise ValueError
            else:
                hists = {float(i - 1): hist for i, hist in enumerate(hists)}
        self.hists = hists
        self.alpha = alpha
        obs = list(hists.values())[0].space
        if extended is None:  # TODO: yields?
            extended = all(hist.is_extended for hist in hists.values())
            if extended:
                extended = list(hists.values())[0].get_yield()
        super().__init__(obs=obs, extended=extended, norm=norm, params={'alpha': alpha},
                         name="LinearMorphing")

    def _counts(self, x, norm):
        densities = [hist.counts(x, norm=norm) for hist in self.hists.values()]
        shape = tf.shape(densities[0])
        densities_flat = [znp.reshape(density, [-1]) for density in densities]
        densities_flat = znp.stack(densities_flat, axis=0)

        # centers = self.hists[0].axes.centers[0][None, :, None]  # TODO: only 1 dim now
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)[None, :, None]
        alpha_shaped = znp.reshape(self.params['alpha'], [1, -1, 1])
        y_flat = tfa.image.interpolate_spline(
            train_points=alphas,
            train_values=densities_flat[None, ...],
            query_points=alpha_shaped,
            order=2

        )
        y_flat = y_flat[0, 0]
        y = tf.reshape(y_flat, shape)
        return y

    def _rel_counts(self, x, norm):
        densities = [hist.rel_counts(x, norm=norm) for hist in self.hists.values()]
        shape = tf.shape(densities[0])
        densities_flat = [znp.reshape(density, [-1]) for density in densities]
        densities_flat = znp.stack(densities_flat, axis=0)

        # centers = self.hists[0].axes.centers[0][None, :, None]  # TODO: only 1 dim now
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)[None, :, None]
        alpha_shaped = znp.reshape(self.params['alpha'], [1, -1, 1])
        y_flat = tfa.image.interpolate_spline(
            train_points=alphas,
            train_values=densities_flat[None, ...],
            query_points=alpha_shaped,
            order=2

        )
        y_flat = y_flat[0, 0]
        y = tf.reshape(y_flat, shape)
        return y

    def _ext_pdf(self, x, norm):
        densities = [hist.ext_pdf(x, norm=norm) for hist in self.hists.values()]
        shape = tf.shape(densities[0])
        densities_flat = [znp.reshape(density, [-1]) for density in densities]
        densities_flat = znp.stack(densities_flat, axis=0)

        # centers = self.hists[0].axes.centers[0][None, :, None]  # TODO: only 1 dim now
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)[None, :, None]
        alpha_shaped = znp.reshape(self.params['alpha'], [1, -1, 1])
        y_flat = tfa.image.interpolate_spline(
            train_points=alphas,
            train_values=densities_flat[None, ...],
            query_points=alpha_shaped,
            order=2

        )
        y_flat = y_flat[0, 0]
        y = tf.reshape(y_flat, shape)
        return y

    def _pdf(self, x, norm):
        densities = [hist.pdf(x, norm=norm) for hist in self.hists.values()]
        shape = tf.shape(densities[0])
        densities_flat = [znp.reshape(density, [-1]) for density in densities]
        densities_flat = znp.stack(densities_flat, axis=0)

        # centers = self.hists[0].axes.centers[0][None, :, None]  # TODO: only 1 dim now
        alphas = znp.array(list(self.hists.keys()), dtype=znp.float64)[None, :, None]
        alpha_shaped = znp.reshape(self.params['alpha'], [1, -1, 1])
        y_flat = tfa.image.interpolate_spline(
            train_points=alphas,
            train_values=densities_flat[None, ...],
            query_points=alpha_shaped,
            order=2

        )
        y_flat = y_flat[0, 0]
        y = tf.reshape(y_flat, shape)
        return y

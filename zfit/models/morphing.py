#  Copyright (c) 2021 zfit
import tensorflow as tf
import tensorflow_addons as tfa

import zfit.z.numpy as znp
from zfit.core.binnedpdf import BaseBinnedPDF


class LinearMorphing(BaseBinnedPDF):
    def __init__(self, alpha, hists, extended=None, norm=None):
        obs = hists[0].obs
        if len(hists) != 3:
            raise ValueError
        self.hists = hists
        self.alpha = alpha
        super().__init__(obs=obs, extended=extended, norm=norm, params={'alpha': alpha},
                         name="LinearMorphing")

    def _ext_pdf(self, x, norm):
        densities = [hist.ext_pdf(hist.space, norm=hist.space) for hist in self.hists]
        shape = tf.shape(densities[0])
        densities_flat = [znp.reshape(density, [-1]) for density in densities]
        densities_flat = znp.stack(densities_flat, axis=0)

        # centers = self.hists[0].axes.centers[0][None, :, None]  # TODO: only 1 dim now
        alphas = znp.array([-1, 0, 1], dtype=znp.float64)[None, :, None]
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

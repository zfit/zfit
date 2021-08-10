#  Copyright (c) 2021 zfit
import numpy as np
import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from ..core.binnedpdf import BaseBinnedPDF
from ..core.interfaces import ZfitData


class BinnedTemplatePDF(BaseBinnedPDF):

    def __init__(self, data, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF"):
        obs = data.space
        if sysshape is None:
            sysshape = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
                        range(data.values().shape.num_elements())}
        params = {}
        params.update(sysshape)
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm)

        self._data = data

    def _ext_pdf(self, x, norm):
        counts = self._ext_integrate(None, None)
        if not isinstance(x, ZfitData):
            return counts
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    def _ext_integrate(self, limits, norm):

        sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
        counts = self._data.values()
        sysshape = tf.reshape(sysshape_flat, counts.shape)
        return counts * sysshape
        #
        # values = self._ext_pdf(None, norm)
        # areas = znp.prod(self._data.axes.widths, axis=0)
        # counts = values * areas
        # return counts

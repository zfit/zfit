#  Copyright (c) 2021 zfit
import tensorflow as tf

import zfit

from .. import convert_to_parameter
from ..core.binnedpdf import BaseBinnedPDF
from ..util.exception import WorkInProgressError


class BinnedTemplatePDF(BaseBinnedPDF):

    def __init__(self, data, sysshape=None, name="BinnedTemplatePDF"):
        obs = data.space
        if sysshape is None:
            sysshape = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
                        range(data.values().shape.num_elements())}
        params = {}
        params.update(sysshape)
        super().__init__(obs=obs, name=name, params=params)

        self._data = data

    def _ext_pdf(self, x, norm_range):
        sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
        counts = self._data.values()
        sysshape = tf.reshape(sysshape_flat, counts.shape)
        return counts * sysshape

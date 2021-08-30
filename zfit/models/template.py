#  Copyright (c) 2021 zfit
import numpy as np
import tensorflow as tf
import zfit.param

from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.interfaces import ZfitData
from ..core.pdf import PDF
from ..core.space import supports
from ..z import numpy as znp


class BinnedTemplatePDFV1(BaseBinnedPDFV1):

    def __init__(self, data, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF"):
        obs = data.space
        if sysshape is None:
            import zfit
            sysshape = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
                        range(data.values().shape.num_elements())}
        params = {}
        params.update(sysshape)
        if extended is None:
            extended = znp.sum(data.values())
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm)

        self._data = data

    def _ext_pdf(self, x, norm):
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    def _pdf(self, x, norm):
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm='same')
    def _counts(self, x, norm=None):
        if norm not in (False, None, self.norm):
            return super()._counts(x, norm)

        sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
        counts = self._data.values()
        sysshape = tf.reshape(sysshape_flat, counts.shape)
        return counts * sysshape

    def _rel_counts(self, x, norm=None):
        if norm not in (False, None, self.norm):
            return super()._rel_counts(x, norm)
        sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
        counts = self._data.values()
        sysshape = tf.reshape(sysshape_flat, counts.shape)
        values = counts * sysshape
        return values / znp.sum(values)

        #
        # values = self._ext_pdf(None, norm)
        # areas = znp.prod(self._data.axes.widths, axis=0)
        # counts = values * areas
        # return counts

# class BinnedTemplatePDF(PDF):
#
#     def __init__(self, data, sysshape=None, extended=None, norm=None, label="BinnedTemplatePDF"):
#         space = data.space
#         if sysshape is None:
#             sysshape = {f'sysshape_{i}': zfit.param.Parameter(f'auto_sysshape_{self}_{i}', 1.)
#                         for i in range(data.values().shape.num_elements())}
#         var = {f'axis_{i}': axis for i, axis in enumerate(space)}
#         var.update(sysshape)
#         super().__init__(var=var, label=label, extended=extended, norm=norm)
#
#         self.sysshape = sysshape
#         self.data = data
#
#     def _ext_pdf(self, var, norm):
#         counts = self._ext_integrate(var, norm)
#         # if not isinstance(x, ZfitData):
#         #     return counts
#         areas = np.prod(self.data.axes.widths, axis=0)
#         density = counts / areas
#         return density
#
#     def _ext_integrate(self, var, norm):
#         counts = self.data.values()
#         if self.sysshape is not None:
#             sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
#             sysshape = tf.reshape(sysshape_flat, counts.shape)
#             counts = counts * sysshape
#         if self.space == var.space and self.space.is_binned \
#                 and (not norm.space or norm.space == self.space):
#             return counts

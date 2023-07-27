#  Copyright (c) 2023 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..core.binnedpdf import BaseBinnedPDFV1
from ..core.space import supports
from ..util.exception import SpecificFunctionNotImplemented
from ..z import numpy as znp


class BinnedTemplatePDFV1(BaseBinnedPDFV1):
    def __init__(
        self, data, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF"
    ):
        # TODO: use scalemodifier instead
        obs = data.space
        if extended is None:
            extended = True
        if sysshape is None:
            sysshape = {}
        if sysshape is True:
            import zfit

            sysshape = {
                f"sysshape_{i}": zfit.Parameter(f"auto_sysshape_{self}_{i}", 1.0)
                for i in range(data.values().shape.num_elements())
            }
        params = {}
        params.update(sysshape)
        self._template_sysshape = sysshape
        if extended is True:
            self._automatically_extended = True
            if sysshape:
                import zfit

                def sumfunc(params):
                    values = self._data.values()
                    sysshape = list(params.values())
                    if sysshape:
                        sysshape_flat = tf.stack(sysshape)
                        sysshape = tf.reshape(sysshape_flat, values.shape)
                        values = values * sysshape
                    return znp.sum(values)

                from zfit.core.parameter import get_auto_number

                extended = zfit.ComposedParameter(
                    f"TODO_name_selfmade_{get_auto_number()}", sumfunc, params=sysshape
                )

            else:
                extended = znp.sum(data.values())
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )

        self._data = data

    def _ext_pdf(self, x, norm):
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm=False)
    def _pdf(self, x, norm):
        counts = self._rel_counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        density = counts / areas
        return density

    @supports(norm="norm")
    # @supports(norm=False)
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._data.values()
        sysshape = list(self._template_sysshape.values())
        if sysshape:
            sysshape_flat = tf.stack(sysshape)
            sysshape = tf.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values

    @supports(norm="norm")
    def _rel_counts(self, x, norm=None):
        values = self._data.values()
        sysshape = list(self._template_sysshape.values())
        if sysshape:
            sysshape_flat = tf.stack(sysshape)
            sysshape = tf.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values / znp.sum(values)


# class BinnedSystematicsPDFV1(FunctorMixin, BaseBinnedPDFV1):
#
#     def __init__(self, pdf, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF"):
#         obs = data.space
#         if sysshape is None:
#             import zfit
#             sysshape = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
#                         range(data.values().shape.num_elements())}
#         params = {}
#         params.update(sysshape)
#         if extended is None:
#             extended = znp.sum(data.values())
#         super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm)
#
#         self._data = data
#
#     def _ext_pdf(self, x, norm):
#         counts = self._counts(x, norm)
#         areas = np.prod(self._data.axes.widths, axis=0)
#         density = counts / areas
#         return density
#
#     def _pdf(self, x, norm):
#         counts = self._counts(x, norm)
#         areas = np.prod(self._data.axes.widths, axis=0)
#         density = counts / areas
#         return density
#
#     @supports(norm='norm')
#     # @supports(norm=False)
#     def _counts(self, x, norm=None):
#
#         sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
#         counts = self._data.values()
#         sysshape = tf.reshape(sysshape_flat, counts.shape)
#         return counts * sysshape
#
#     @supports(norm='norm')
#     def _rel_counts(self, x, norm=None):
#         sysshape_flat = tf.stack([p for name, p in self.params.items() if name.startswith('sysshape')])
#         counts = self._data.values()
#         sysshape = tf.reshape(sysshape_flat, counts.shape)
#         values = counts * sysshape
#         return values / znp.sum(values)
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

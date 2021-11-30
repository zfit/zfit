#  Copyright (c) 2021 zfit
from typing import Mapping

import numpy as np

from .binned_functor import BaseBinnedFunctorPDF
from ..core.space import supports
from ..core.interfaces import ZfitBinnedPDF
import tensorflow as tf
import zfit.z.numpy as znp
from ..util.exception import SpecificFunctionNotImplemented


class BinwiseModifier(BaseBinnedFunctorPDF):

    def __init__(self, pdf, modifiers=None, extended=None, norm=None, name="BinnedTemplatePDF"):
        obs = pdf.space
        if not isinstance(pdf, ZfitBinnedPDF):
            raise TypeError("pdf must be a BinnedPDF")
        if extended is None:
            extended = pdf.is_extended
        if modifiers is None:
            modifiers = True
        if modifiers is True:
            import zfit
            modifiers = {f'sysshape_{i}': zfit.Parameter(f'auto_sysshape_{self}_{i}', 1.) for i in
                         range(pdf.counts(obs).shape.num_elements())}
        if not isinstance(modifiers, dict):
            raise TypeError("modifiers must be a dict-like object or True or None")
        params = modifiers.copy()
        self._binwise_modifiers = modifiers
        if extended is True:
            self._automatically_extended = True
            if modifiers:
                import zfit

                def sumfunc(params):
                    values = self.pdfs[0].counts(obs)
                    sysshape = list(params.values())
                    if sysshape:
                        sysshape_flat = tf.stack(sysshape)
                        sysshape = znp.reshape(sysshape_flat, values.shape)
                        values = values * sysshape
                    return znp.sum(values)

                from zfit.core.parameter import get_auto_number
                extended = zfit.ComposedParameter(f'AUTO_binwise_modifier_{get_auto_number()}', sumfunc,
                                                  params=modifiers)

            else:
                extended = self.pdfs[0].get_yield()
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(obs=obs, name=name, params=params, models=pdf, extended=extended, norm=norm)

    @supports(norm=True)
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._counts_with_modifiers(x, norm)
        return values

    def _counts_with_modifiers(self, x, norm):
        values = self.pdfs[0].counts(x, norm=norm)
        modifiers = list(self._binwise_modifiers.values())
        if modifiers:
            sysshape_flat = tf.stack(modifiers)
            modifiers = znp.reshape(sysshape_flat, values.shape)
            values = values * modifiers
        return values

    @supports(norm='space')
    def _rel_counts(self, x, norm=None):
        values = self._counts_with_modifiers(x, norm)
        return values / znp.sum(values)

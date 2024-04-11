#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # ruff: noqa: F401

from collections.abc import Mapping

import tensorflow as tf

import zfit.z.numpy as znp

from ..core.interfaces import ZfitBinnedPDF
from ..core.space import supports
from ..util import ztyping
from ..util.exception import SpecificFunctionNotImplemented
from .binned_functor import BaseBinnedFunctorPDF


class BinwiseScaleModifier(BaseBinnedFunctorPDF):
    def __init__(
        self,
        pdf: ZfitBinnedPDF,
        modifiers: bool | Mapping[str, ztyping.ParamTypeInput] | None = None,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormInputType = None,
        name: str | None = "BinnedTemplatePDF",
        label: str | None = None,
    ) -> None:
        """Modifier that scales each bin separately of the *pdf*.

        Binwise modification can be used to account for uncorrelated or correlated uncertainties.

        Args:
            pdf: Binned pdf to be modified.
            modifiers: Modifiers for each bin.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        obs = pdf.space
        if not isinstance(pdf, ZfitBinnedPDF):
            msg = "pdf must be a BinnedPDF"
            raise TypeError(msg)
        if extended is None:
            extended = pdf.is_extended
        if modifiers is None:
            modifiers = True
        if modifiers is True:
            import zfit

            modifiers = {
                f"sysshape_{i}": zfit.Parameter(f"auto_sysshape_{self}_{i}", 1.0)
                for i in range(pdf.counts(obs).shape.num_elements())
            }
        if not isinstance(modifiers, dict):
            msg = "modifiers must be a dict-like object or True or None"
            raise TypeError(msg)
        params = modifiers.copy()
        self._binwise_modifiers = modifiers
        if extended is True:
            self._automatically_extended = True
            if modifiers:
                import zfit

                def sumfunc(params):
                    del params  # unused
                    values = self.counts()
                    return znp.sum(values)

                from zfit.core.parameter import get_auto_number

                params_sumfunc = modifiers.copy()
                dep_params = {}
                for p in pdf.get_params():
                    while True:
                        i = get_auto_number()
                        name_tmp = f"DUMMPY_PARAM_{i}"
                        if name_tmp not in params_sumfunc:
                            dep_params[name_tmp] = p
                            break

                params_sumfunc.update(dep_params)
                extended = zfit.ComposedParameter(
                    f"AUTO_binwise_modifier_{get_auto_number()}",
                    sumfunc,
                    params=params_sumfunc,
                )

            else:
                extended = pdf.get_yield()
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(obs=obs, name=name, params=params, models=pdf, extended=extended, norm=norm, label=label)

    @supports(norm=True)
    def _counts(self, x, norm=None):
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        return self._counts_with_modifiers(x, norm)

    def _counts_with_modifiers(self, x, norm):
        values = self.pdfs[0].counts(x, norm=norm)
        if modifiers := list(self._binwise_modifiers.values()):
            sysshape_flat = tf.stack(modifiers)
            modifiers = znp.reshape(sysshape_flat, values.shape)
            values = values * modifiers
        return values

    @supports(norm="space")
    def _rel_counts(self, x, norm=None):
        values = self._counts_with_modifiers(x, norm)
        return values / znp.sum(values)

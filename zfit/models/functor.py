"""
Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the models easier.

Their implementation is often non-trivial.
"""
#  Copyright (c) 2019 zfit

from collections import OrderedDict
import itertools
from typing import Union, List, Optional

import tensorflow as tf
import numpy as np

from zfit import ztf
from ..core.interfaces import ZfitPDF, ZfitModel, ZfitSpace
from ..core.limits import no_norm_range, supports
from ..core.basepdf import BasePDF
from ..core.parameter import Parameter, convert_to_parameter
from ..models.basefunctor import FunctorMixin
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (ExtendedPDFError, AlreadyExtendedPDFError, AxesNotUnambiguousError,
                              LimitsOverdefinedError,
                              ModelIncompatibleError, )
from ..util.temporary import TemporarilySet
from ..settings import ztypes, run


class BaseFunctor(FunctorMixin, BasePDF):

    def __init__(self, pdfs, name="BaseFunctor", **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(models=self.pdfs, name=name, **kwargs)
        self._set_norm_range_from_daugthers()
        self._component_norm_range_holder = None

    def _get_component_norm_range(self):
        return self._component_norm_range_holder

    def _set_component_norm_range(self, norm_range: ztyping.LimitsTypeInput):
        norm_range = self._check_input_norm_range(norm_range=norm_range)
        if norm_range.limits in (False, None):
            if self._get_component_norm_range() is None:
                raise RuntimeError("Cannot use `False` as `norm_range` without previously setting the "
                                   "`component_norm_range`.")

        def setter(value):
            self._component_norm_range_holder = value

        return TemporarilySet(value=norm_range, setter=setter, getter=self._get_component_norm_range)

    def _set_norm_range_from_daugthers(self):
        norm_range = super().norm_range
        if norm_range.limits is None:
            norm_range_candidat = self._infer_norm_range_from_daughters()
            # if norm_range_candidat is False:
            #     raise LimitsOverdefinedError("Daughter pdfs do not agree on a `norm_range` and no `norm_range`"
            #                                  "has been explicitly set.")
            if isinstance(norm_range_candidat, ZfitSpace):  # TODO(Mayou36, #77): different obs?
                norm_range = norm_range_candidat

        self._norm_range = norm_range

    def _infer_norm_range_from_daughters(self):
        norm_ranges = set(model.norm_range for model in self.models)
        obs = set(norm_range.obs for norm_range in norm_ranges)
        if len(norm_ranges) == 1:
            return norm_ranges.pop()
        elif len(obs) > 1:  # TODO(Mayou36, #77): different obs?
            return None
        else:
            return False

    def _single_hook_unnormalized_pdf(self, x, component_norm_range, name):
        if component_norm_range.limits is not None:
            with self._set_component_norm_range(norm_range=component_norm_range):
                return super()._single_hook_unnormalized_pdf(x, component_norm_range, name)
        else:
            return super()._single_hook_unnormalized_pdf(x, component_norm_range, name)

    def _single_hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_integrate(limits, norm_range, name)

    def _single_hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_analytic_integrate(limits, norm_range, name)

    def _single_hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_numeric_integrate(limits, norm_range, name)

    def _single_hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_partial_integrate(x, limits, norm_range, name)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range, name='_hook_partial_analytic_integrate'):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_partial_analytic_integrate(x, limits, norm_range, name)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range, name='_hook_partial_numeric_integrate'):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_partial_numeric_integrate(x, limits, norm_range, name)

    def _single_hook_normalization(self, limits, name="_hook_normalization"):
        with self._set_component_norm_range(norm_range=limits):
            return super()._single_hook_normalization(limits, name)

    def _single_hook_pdf(self, x, norm_range, name="_hook_pdf"):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_pdf(x, norm_range, name)

    def _single_hook_log_pdf(self, x, norm_range, name):
        with self._set_component_norm_range(norm_range=norm_range):
            return super()._single_hook_log_pdf(x, norm_range, name)

    def _single_hook_sample(self, n, limits, name):
        with self._set_component_norm_range(norm_range=limits):
            return super()._single_hook_sample(n, limits, name)

    @property
    def pdfs_extended(self):
        return [pdf.is_extended for pdf in self.pdfs]

    @property
    def _models(self) -> List[ZfitModel]:
        return self.pdfs


class SumPDF(BaseFunctor):

    def __init__(self, pdfs: List[ZfitPDF], fracs: Optional[ztyping.ParamTypeInput] = None,
                 obs: ztyping.ObsTypeInput = None,
                 name: str = "SumPDF"):
        """Create the sum of the `pdfs` with `fracs` as coefficients.

        Args:
            pdfs (pdf): The pdfs to add.
            fracs (iterable): coefficients for the linear combination of the pdfs. If pdfs are
                extended, this throws an error.

                  - len(frac) == len(basic) - 1 results in the interpretation of a non-extended pdf.
                    The last coefficient will equal to 1 - sum(frac)
                  - len(frac) == len(pdf) each pdf in `pdfs` will become an extended pdf with the
                    given yield.
            name (str):
        """
        # Check user input, improve TODO
        self._fracs = None

        set_yield_at_end = False
        pdfs = convert_to_container(pdfs)
        self.pdfs = pdfs
        if len(pdfs) < 2:
            raise ValueError("Cannot build a sum of a single pdf")
        if fracs is not None:
            fracs = convert_to_container(fracs)
            fracs = [convert_to_parameter(frac) for frac in fracs]

        # check if all extended
        extended_pdfs = self.pdfs_extended
        implicit = None
        extended = None
        if all(extended_pdfs):
            implicit = True
            extended = True

        # all extended except one -> fraction
        elif sum(extended_pdfs) == len(extended_pdfs) - 1:
            implicit = True
            extended = False

        # no pdf is extended -> using `fracs`
        elif not any(extended_pdfs) and fracs is not None:
            # make extended
            if len(fracs) == len(pdfs):
                implicit = False
                extended = True
            elif len(fracs) == len(pdfs) - 1:
                implicit = False
                extended = False

        # catch if args don't fit known case
        value_error = implicit is None or extended is None
        if (implicit and fracs is not None) or value_error:
            raise ModelIncompatibleError("Wrong arguments. Either"
                                         "\n a) `pdfs` are not extended and `fracs` is given with length pdfs "
                                         "(-> pdfs get extended) or pdfs - 1 (fractions)"
                                         "\n b) all or all except 1 `pdfs` are extended and fracs is None.")

        # create fracs if one is not extended
        if not extended and implicit:
            fracs = []
            not_extended_position = None
            new_pdfs = []
            for i, pdf in enumerate(pdfs):
                if pdf.is_extended:
                    fracs.append(pdf.get_yield())
                    pdf = pdf.copy()
                    pdf._set_yield_inplace(None)  # make non-extended

                else:
                    fracs.append(tf.constant(0., dtype=ztypes.float))
                    not_extended_position = i
                new_pdfs.append(pdf)
            pdfs = new_pdfs
            remaining_frac = tf.constant(1., dtype=ztypes.float) - tf.add_n(fracs)
            if run.numeric_checks:
                assert_op = tf.Assert(tf.greater_equal(remaining_frac, tf.constant(0., dtype=ztypes.float)),
                                      data=[remaining_frac])  # check fractions
                deps = [assert_op]
            else:
                deps = []
            with tf.control_dependencies(deps):
                # TODO(Mayou36): always last position?
                fracs[not_extended_position] = tf.identity(remaining_frac)
            implicit = False  # now it's explicit

        elif not extended and not implicit:
            remaining_frac = tf.constant(1., dtype=ztypes.float) - tf.add_n(fracs)
            if run.numeric_checks:
                assert_op = tf.Assert(tf.greater_equal(remaining_frac, tf.constant(0., dtype=ztypes.float)),
                                      data=[remaining_frac])  # check fractions
                deps = [assert_op]
            else:
                deps = []
            with tf.control_dependencies(deps):
                fracs.append(tf.identity(remaining_frac))

        # make extended
        elif extended and not implicit:
            yields = fracs
            pdfs = [pdf.create_extended(yield_) for pdf, yield_ in zip(pdfs, yields)]

            implicit = True

        elif extended and implicit:
            yields = [pdf.get_yield() for pdf in pdfs]

        if extended:
            # TODO(Mayou36): convert to correct dtype
            sum_yields = tf.reduce_sum([tf.convert_to_tensor(y, preferred_dtype=ztypes.float) for y in yields])
            yield_fracs = [yield_ / sum_yields for yield_ in yields]
            self.fracs = yield_fracs
            # self.fracs = yield_fracs
            set_yield_at_end = True
            self._maybe_extended_fracs = [tf.constant(1, dtype=ztypes.float)] * len(self.pdfs)
        else:
            self._maybe_extended_fracs = fracs

        self.pdfs = pdfs

        params = OrderedDict()
        for i, frac in enumerate(self._maybe_extended_fracs):
            params['frac_{}'.format(i)] = frac

        super().__init__(pdfs=pdfs, obs=obs, params=params, name=name)
        if set_yield_at_end:
            self._set_yield_inplace(sum_yields)

    @property
    def fracs(self):
        fracs = self._fracs
        if fracs is None:
            fracs = self._maybe_extended_fracs
        return fracs

    @fracs.setter
    def fracs(self, value):
        self._fracs = value

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm_range=norm_range, log=log)

    def _unnormalized_pdf(self, x):
        norm_range = self._get_component_norm_range()
        return self._pdf(x=x, norm_range=norm_range)
        # raise NotImplementedError
        # pdfs = self.pdfs
        # fracs = self.fracs
        # func = tf.accumulate_n(
        #     [scale * pdf.unnormalized_pdf(x) for pdf, scale in zip(pdfs, fracs)])
        # return func

    def _pdf(self, x, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        prob = tf.add_n([pdf.pdf(x, norm_range=norm_range) * frac for pdf, frac in zip(pdfs, fracs)])
        # prob = tf.accumulate_n([pdf.pdf(x, norm_range=norm_range) * scale for pdf, scale in zip(pdfs, fracs)])
        return prob

    def _set_yield(self, value: Union[Parameter, None]):
        # TODO: what happens now with the daughters?
        if all(self.pdfs_extended) and self.is_extended and value is not None:  # to be able to set the yield in the
            # beginning
            raise AlreadyExtendedPDFError("Cannot set the yield of a PDF with extended daughters.")
        elif all(self.pdfs_extended) and self.is_extended and value is None:  # not extended anymore
            reciprocal_yield = tf.reciprocal(self.get_yield())
            self._maybe_extended_fracs = [reciprocal_yield] * len(self._maybe_extended_fracs)
        else:
            super()._set_yield(value=value)

    @supports(norm_range=True, multiple_limits=True)
    def _integrate(self, limits, norm_range):
        pdfs = self.pdfs
        fracs = self._maybe_extended_fracs
        assert norm_range not in (None, False), "Bug, who requested an unnormalized integral?"
        integrals = [pdf.integrate(limits=limits, norm_range=norm_range) for pdf in pdfs]
        integrals = [integral * frac for integral, frac in zip(integrals, fracs)]
        integral = tf.reduce_sum(integrals)
        return integral

    @supports(norm_range=True, multiple_limits=True)
    def _analytic_integrate(self, limits, norm_range):
        pdfs = self.pdfs
        fracs = self._maybe_extended_fracs
        assert norm_range not in (None, False), "Bug, who requested an unnormalized integral?"
        try:
            integrals = [pdf.analytic_integrate(limits=limits, norm_range=norm_range) for pdf in pdfs]
        except NotImplementedError as original_error:
            raise NotImplementedError("analytic_integrate of pdf {name} is not implemented in this"
                                      " SumPDF, as at least one sub-pdf does not implement it."
                                      "Original message:\n{error}".format(name=self.name,
                                                                          error=original_error))

        integrals = [integral * frac for integral, frac in zip(integrals, fracs)]
        integral = tf.reduce_sum(integrals)
        return integral

    @supports(norm_range=True, multiple_limits=True)
    def _partial_integrate(self, x, limits, norm_range):
        raise RuntimeError("Currently not available, cleanup with yields expected.")

    # @supports()
    # def _partial_analytic_integrate(self, x, limits, norm_range):
    #     pdfs = self.pdfs
    #     frac = self.fracs
    #     try:
    #         partial_integral = [pdf.analytic_integrate(limits=limits, norm_range=norm_range) for pdf in pdfs]
    #     except NotImplementedError as original_error:
    #         raise NotImplementedError("partial_analytic_integrate of pdf {name} is not implemented in this"
    #                                   " SumPDF, as at least one sub-pdf does not implement it."
    #                                   "Original message:\n{error}".format(name=self.name,
    #                                                                       error=original_error))
    #     partial_integral = tf.stack([partial_integral * s for pdf, s in zip(partial_integral, frac)])
    #     partial_integral = tf.reduce_sum(partial_integral, axis=0)
    #     return partial_integral


class ProductPDF(BaseFunctor):  # TODO: unfinished
    def __init__(self, pdfs: List[ZfitPDF], obs: ztyping.ObsTypeInput = None, name="ProductPDF"):
        super().__init__(pdfs=pdfs, obs=obs, name=name)

    def _unnormalized_pdf(self, x: ztyping.XType):

        norm_range = self._get_component_norm_range()
        return np.prod([pdf.unnormalized_pdf(x, component_norm_range=norm_range.get_subspace(obs=pdf.obs))
                        for pdf in self.pdfs])

    def _pdf(self, x, norm_range):
        if all(not dep for dep in self._model_same_obs):

            probs = [pdf.pdf(x=x, norm_range=norm_range.get_subspace(obs=pdf.obs)) for pdf in self.pdfs]
            return tf.reduce_prod(probs, axis=0)
        else:
            raise NotImplementedError

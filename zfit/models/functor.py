"""Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the models easier.

Their implementation is often non-trivial.
"""
#  Copyright (c) 2021 zfit
import functools
import operator
from collections import OrderedDict
from typing import List, Optional

import tensorflow as tf

import zfit.z.numpy as znp

from .. import z
from ..core.basepdf import BasePDF
from ..core.coordinates import convert_to_obs_str
from ..core.interfaces import ZfitData, ZfitModel, ZfitPDF
from ..core.parameter import convert_to_parameter
from ..core.space import supports
from ..models.basefunctor import FunctorMixin, extract_daughter_input_obs
from ..settings import run, ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (AnalyticIntegralNotImplemented,
                              ModelIncompatibleError,
                              NormRangeUnderdefinedError, ObsIncompatibleError,
                              SpecificFunctionNotImplemented)
from ..util.warnings import warn_advanced_feature, warn_changed_feature
from ..z.random import counts_multinomial


class BaseFunctor(FunctorMixin, BasePDF):

    def __init__(self, pdfs, name="BaseFunctor", **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(models=self.pdfs, name=name, **kwargs)
        self._set_norm_range_from_daugthers()

    def _set_norm_range_from_daugthers(self):
        norm_range = super().norm_range
        if not norm_range.limits_are_set:
            norm_range = extract_daughter_input_obs(obs=norm_range,
                                                    spaces=[model.space for model in self.models])
        if not norm_range.limits_are_set:
            raise NormRangeUnderdefinedError(
                f"Daughter pdfs {self.pdfs} do not agree on a `norm_range` and/or no `norm_range`"
                "has been explicitly set.")

        self.set_norm_range(norm_range)

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
        """Create the sum of the `pdfs` with `fracs` as coefficients or the yields, if extended pdfs are given.

        If *all* pdfs are extended, the fracs is optional and the (normalized) yields will be used as fracs.
        If fracs is given, this will be used as the fractions, regardless of whether the pdfs have a yield or not.

        The parameters of the SumPDF are the fractions that are used to multiply the output of each daughter pdf.
        They can be accessed with `pdf.params` and have names f"frac_{i}" with i starting from 0 and going to the number
        of pdfs given.

        To get the component outputs of this pdf, e.g. to plot it, use `pdf.params.values()` to iterate through the
        fracs and `pdfs` to get the pdfs. For example

        .. code-block:: python

            for pdf, frac in zip(sumpdf.pdfs, sumpdf.params.values()):
                frac_integral = pdf.integrate(...) * frac

        Args:
            pdfs: The pdfs to be added.
            fracs: Coefficients for the linear combination of the pdfs. Optional if *all* pdfs are extended.

                - len(frac) == len(basic) - 1 results in the interpretation of a non-extended pdf.
                  The last coefficient will equal to 1 - sum(frac)
                - len(frac) == len(pdf): the fracs will be used as is and no normalization attempt is taken.
            name: |name_arg_descr|

        Raises
            ModelIncompatibleError: If model is incompatible.
        """
        # Check user input
        self._fracs = None

        pdfs = convert_to_container(pdfs)
        self.pdfs = pdfs
        if len(pdfs) < 2:
            raise ValueError(f"Cannot build a sum of a single pdf {pdfs}")
        common_obs = obs if obs is not None else pdfs[0].obs
        common_obs = convert_to_obs_str(common_obs)
        if not all(frozenset(pdf.obs) == frozenset(common_obs) for pdf in pdfs):
            raise ObsIncompatibleError("Currently, sums are only supported in the same observables")

        # check if all extended
        are_extended = [pdf.is_extended for pdf in pdfs]
        all_extended = all(are_extended)
        no_extended = not any(are_extended)

        fracs = convert_to_container(fracs)
        if fracs:  # not None or empty list
            fracs = [convert_to_parameter(frac) for frac in fracs]
        elif not all_extended:
            raise ModelIncompatibleError(f"Not all pdf {pdfs} are extended and no fracs {fracs} are provided.")

        if not no_extended and fracs:
            warn_advanced_feature(f"This SumPDF is built with fracs {fracs} and {'all' if all_extended else 'some'} "
                                  f"pdf are extended: {pdfs}."
                                  f" This will ignore the yields of the already extended pdfs and the result will"
                                  f" be a not extended SumPDF.", identifier='sum_extended_frac')

        # catch if args don't fit known case

        if fracs:
            # create fracs if one is missing
            if len(fracs) == len(pdfs) - 1:
                remaining_frac_func = lambda: tf.constant(1., dtype=ztypes.float) - tf.add_n(fracs)
                remaining_frac = convert_to_parameter(remaining_frac_func,
                                                      params=fracs)
                if run.numeric_checks:
                    tf.debugging.assert_non_negative(remaining_frac,
                                                     f"The remaining fraction is negative, the sum of fracs is > 0. Fracs: {fracs}")  # check fractions

                # IMPORTANT! Otherwise, recursion due to namespace capture in the lambda
                fracs_cleaned = fracs + [remaining_frac]

            elif len(fracs) == len(pdfs):
                warn_changed_feature("A SumPDF with the number of fractions equal to the number of pdf will no longer "
                                     "be extended. To make it extended, either manually use 'create_exteneded' or set "
                                     "the yield. OR provide all pdfs as extended pdfs and do not provide a fracs "
                                     "argument.", identifier='new_sum')
                fracs_cleaned = fracs

            else:
                raise ModelIncompatibleError(f"If all PDFs are not extended {pdfs}, the fracs {fracs} have to be of"
                                             f" the same length as pdf or one less.")
            param_fracs = fracs_cleaned

        # for the extended case, take the yields, normalize them, in case no fracs are given.
        if all_extended and not fracs:
            yields = [pdf.get_yield() for pdf in pdfs]

            def sum_yields_func():
                return znp.sum(
                    [tf.convert_to_tensor(value=y, dtype_hint=ztypes.float) for y in yields])

            sum_yields = convert_to_parameter(sum_yields_func, params=yields)
            yield_fracs = [convert_to_parameter(lambda sum_yields, yield_: yield_ / sum_yields,
                                                params=[sum_yields, yield_])
                           for yield_ in yields]

            fracs_cleaned = None
            param_fracs = yield_fracs

        self.pdfs = pdfs
        self._fracs = param_fracs
        self._original_fracs = fracs_cleaned

        params = OrderedDict()
        for i, frac in enumerate(param_fracs):
            params[f'frac_{i}'] = frac

        super().__init__(pdfs=pdfs, obs=obs, params=params, name=name)
        if all_extended and not fracs_cleaned:
            self.set_yield(sum_yields)
            # self.set_yield(sum_yields)  # TODO(SUM): why not the public method below?

    @property
    def fracs(self):
        return self._fracs

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm_range=norm_range, log=log)

    def _unnormalized_pdf(self, x):  # NOT _pdf, as the normalization range can differ
        pdfs = self.pdfs
        fracs = self.params.values()
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)
        return z.convert_to_tensor(prob)

    def _pdf(self, x, norm_range):  # NOT _pdf, as the normalization range can differ
        equal_norm_ranges = len(set([pdf.norm_range for pdf in self.pdfs] + [norm_range])) == 1
        if not equal_norm_ranges:
            raise SpecificFunctionNotImplemented
        pdfs = self.pdfs
        fracs = self.params.values()
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)
        return z.convert_to_tensor(prob)

    @supports(multiple_limits=True)
    def _integrate(self, limits, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        # TODO(SUM): why was this needed?
        # assert norm_range not in (None, False), "Bug, who requested an unnormalized integral?"
        integrals = [frac * pdf.integrate(limits=limits)  # do NOT propagate the norm_range!
                     for pdf, frac in zip(pdfs, fracs)]
        integral = sum(integrals)
        return z.convert_to_tensor(integral)

    @supports(multiple_limits=True)
    def _analytic_integrate(self, limits, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            integrals = [frac * pdf.analytic_integrate(limits=limits)  # do NOT propagate the norm_range!
                         for pdf, frac in zip(pdfs, fracs)]
        except AnalyticIntegralNotImplemented as error:
            raise AnalyticIntegralNotImplemented(
                f"analytic_integrate of pdf {self.name} is not implemented in this"
                f" SumPDF, as at least one sub-pdf does not implement it.") from error

        integral = sum(integrals)
        return z.convert_to_tensor(integral)

    @supports(multiple_limits=True)
    def _partial_integrate(self, x, limits, norm_range):

        pdfs = self.pdfs
        fracs = self.fracs

        partial_integral = [pdf.partial_integrate(x=x, limits=limits) * frac  # do NOT propagate the norm_range!
                            for pdf, frac in zip(pdfs, fracs)]
        partial_integral = sum(partial_integral)
        return z.convert_to_tensor(partial_integral)

    @supports(multiple_limits=True)
    def _partial_analytic_integrate(self, x, limits, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            partial_integral = [pdf.partial_analytic_integrate(x=x, limits=limits) * frac
                                # do NOT propagate the norm_range!
                                for pdf, frac in zip(pdfs, fracs)]
        except AnalyticIntegralNotImplemented as error:
            raise AnalyticIntegralNotImplemented(
                "partial_analytic_integrate of pdf {name} is not implemented in this"
                " SumPDF, as at least one sub-pdf does not implement it.") from error
        partial_integral = sum(partial_integral)
        return z.convert_to_tensor(partial_integral)

    @supports(multiple_limits=True)
    def _sample(self, n, limits):
        if (isinstance(n, str)):
            n = [n] * len(self.pdfs)
        else:
            n = tf.unstack(counts_multinomial(total_count=n, probs=self.fracs), axis=0)

        samples = []
        for pdf, n_sample in zip(self.pdfs, n):
            sub_sample = pdf.sample(n=n_sample, limits=limits)
            if isinstance(sub_sample, ZfitData):
                sub_sample = sub_sample.value()
            samples.append(sub_sample)
        sample = znp.concatenate(samples, axis=0)
        sample = tf.random.shuffle(sample)
        return sample


class ProductPDF(BaseFunctor):  # TODO: compose of smaller Product PDF by disassembling components subsets of obs
    def __init__(self, pdfs: List[ZfitPDF], obs: ztyping.ObsTypeInput = None, name="ProductPDF"):
        super().__init__(pdfs=pdfs, obs=obs, name=name)

    def _unnormalized_pdf(self, x: ztyping.XType):
        probs = [pdf.pdf(x, norm_range=False) for pdf in self.pdfs]
        prob = functools.reduce(operator.mul, probs)
        return z.convert_to_tensor(prob)

    def _pdf(self, x, norm_range):
        equal_norm_ranges = len(set([pdf.norm_range for pdf in self.pdfs] + [norm_range])) == 1  # all equal
        if not any(self._model_same_obs) and equal_norm_ranges:

            probs = [pdf.pdf(x=x) for pdf in self.pdfs]
            prob = functools.reduce(operator.mul, probs)
            return z.convert_to_tensor(prob)
        else:
            raise SpecificFunctionNotImplemented

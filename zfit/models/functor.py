"""Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the models easier.

Their implementation is often non-trivial.
"""
#  Copyright (c) 2023 zfit

from __future__ import annotations

import functools
import operator
from collections import Counter
from collections.abc import Iterable

import tensorflow as tf

import zfit.z.numpy as znp
from .basefunctor import _preprocess_init_sum
from .. import z
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData, ZfitPDF
from ..core.space import supports
from ..models.basefunctor import FunctorMixin, extract_daughter_input_obs
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (
    AnalyticIntegralNotImplemented,
    NormRangeUnderdefinedError,
    SpecificFunctionNotImplemented,
)
from ..z.random import counts_multinomial


# TODO: order of spaces if the obs is different from the wrapped pdf
class BaseFunctor(FunctorMixin, BasePDF):
    def __init__(self, pdfs, name="BaseFunctor", **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(models=self.pdfs, name=name, **kwargs)
        self._set_norm_from_daugthers()

    def _set_norm_from_daugthers(self):
        norm = super().norm
        if not norm.limits_are_set:
            norm = extract_daughter_input_obs(
                obs=norm, spaces=[model.space for model in self.models]
            )
        if not norm.limits_are_set:
            raise NormRangeUnderdefinedError(
                f"Daughter pdfs {self.pdfs} do not agree on a `norm` and/or no `norm`"
                "has been explicitly set."
            )

        self.set_norm_range(norm)

    @property
    def pdfs_extended(self):
        return [pdf.is_extended for pdf in self.pdfs]


class SumPDF(BaseFunctor):
    def __init__(
        self,
        pdfs: Iterable[ZfitPDF],
        fracs: ztyping.ParamTypeInput | None = None,
        obs: ztyping.ObsTypeInput = None,
        name: str = "SumPDF",
    ):
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
        (
            all_extended,
            fracs_cleaned,
            param_fracs,
            params,
            sum_yields,
        ) = _preprocess_init_sum(fracs, obs, pdfs)

        self._fracs = param_fracs
        self._original_fracs = fracs_cleaned

        super().__init__(pdfs=pdfs, obs=obs, params=params, name=name)
        if all_extended and not fracs_cleaned:
            self.set_yield(sum_yields)

    @property
    def fracs(self):
        return self._fracs

    def _apply_yield(self, value: float, norm: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm=norm, log=log)

    def _unnormalized_pdf(self, x):  # NOT _pdf, as the normalization range can differ
        pdfs = self.pdfs
        fracs = self.params.values()
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)
        return z.convert_to_tensor(prob)

    @supports(norm=True, multiple_limits=True)
    def _pdf(self, x, norm):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if norm and not equal_norm_ranges:
            raise SpecificFunctionNotImplemented
        pdfs = self.pdfs
        fracs = self.params.values()
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)
        return z.convert_to_tensor(prob)

    @supports(multiple_limits=True)
    def _integrate(self, limits, norm, options):
        pdfs = self.pdfs
        fracs = self.fracs
        # TODO(SUM): why was this needed?
        # assert norm_range not in (None, False), "Bug, who requested an unnormalized integral?"
        integrals = [
            frac
            * pdf.integrate(
                limits=limits, options=options
            )  # do NOT propagate the norm_range!
            for pdf, frac in zip(pdfs, fracs)
        ]
        return znp.sum(integrals, axis=0)

    @supports(multiple_limits=True)
    def _analytic_integrate(self, limits, norm):
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            integrals = [
                frac
                * pdf.analytic_integrate(
                    limits=limits
                )  # do NOT propagate the norm_range!
                for pdf, frac in zip(pdfs, fracs)
            ]
        except AnalyticIntegralNotImplemented as error:
            raise AnalyticIntegralNotImplemented(
                f"analytic_integrate of pdf {self.name} is not implemented in this"
                f" SumPDF, as at least one sub-pdf does not implement it."
            ) from error

        integral = sum(integrals)
        return z.convert_to_tensor(integral)

    @supports(multiple_limits=True)
    def _partial_integrate(self, x, limits, norm, *, options):
        pdfs = self.pdfs
        fracs = self.fracs

        # do NOT propagate the norm_range!
        partial_integral = [
            pdf.partial_integrate(x=x, limits=limits, options=options) * frac
            for pdf, frac in zip(pdfs, fracs)
        ]
        partial_integral = sum(partial_integral)
        return z.convert_to_tensor(partial_integral)

    @supports(multiple_limits=True)
    def _partial_analytic_integrate(self, x, limits, norm, options):
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            partial_integral = [
                pdf.partial_analytic_integrate(x=x, limits=limits) * frac
                # do NOT propagate the norm_range!
                for pdf, frac in zip(pdfs, fracs)
            ]
        except AnalyticIntegralNotImplemented as error:
            raise AnalyticIntegralNotImplemented(
                "partial_analytic_integrate of pdf {name} is not implemented in this"
                " SumPDF, as at least one sub-pdf does not implement it."
            ) from error
        partial_integral = sum(partial_integral)
        return z.convert_to_tensor(partial_integral)

    @supports(multiple_limits=True)
    def _sample(self, n, limits):
        if isinstance(n, str):
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
        sample = z.random.shuffle(sample)
        return sample


class ProductPDF(BaseFunctor):
    def __init__(
        self, pdfs: list[ZfitPDF], obs: ztyping.ObsTypeInput = None, name="ProductPDF"
    ):
        super().__init__(pdfs=pdfs, obs=obs, name=name)

        same_obs_pdfs = []
        disjoint_obs_pdfs = []
        same_obs = Counter([ob for pdf in self.pdfs for ob in pdf.obs])
        same_obs = {k for k, v in same_obs.items() if v > 1}
        for pdf in self.pdfs:
            if set(pdf.obs).isdisjoint(same_obs):
                disjoint_obs_pdfs.append(pdf)
            else:
                same_obs_pdfs.append(pdf)
        if same_obs_pdfs and disjoint_obs_pdfs:
            same_obs_pdf = type(self)(pdfs=same_obs_pdfs)
            disjoint_obs_pdfs.append(same_obs_pdf)
            self.add_cache_deps(same_obs_pdfs)
            self._prod_is_same_obs_pdf = False
            self._prod_disjoint_obs_pdfs = disjoint_obs_pdfs
        elif disjoint_obs_pdfs:
            self._prod_is_same_obs_pdf = False
            self._prod_disjoint_obs_pdfs = disjoint_obs_pdfs
            assert set(disjoint_obs_pdfs) == set(self.pdfs)
        else:
            self._prod_is_same_obs_pdf = True
            self._prod_disjoint_obs_pdfs = None

    def _unnormalized_pdf(self, x: ztyping.XType):
        probs = [pdf.pdf(x, norm=False) for pdf in self.pdfs]
        prob = functools.reduce(operator.mul, probs)
        return z.convert_to_tensor(prob)

    @supports(norm=True, multiple_limits=True)
    def _pdf(self, x, norm):
        equal_norm_ranges = (
            len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        )  # all equal
        if not self._prod_is_same_obs_pdf and equal_norm_ranges:
            probs = [pdf.pdf(var=x, norm=norm) for pdf in self._prod_disjoint_obs_pdfs]
            prob = functools.reduce(operator.mul, probs)
            return z.convert_to_tensor(prob)
        else:
            raise SpecificFunctionNotImplemented

    @supports(norm=False)
    def _integrate(self, limits, norm, options):
        if not self._prod_is_same_obs_pdf:
            integrals = []
            for pdf in self._prod_disjoint_obs_pdfs:
                limit = limits.with_obs(pdf.obs)
                integrals.append(
                    pdf.integrate(limits=limit, norm=norm, options=options)
                )
            integral = functools.reduce(operator.mul, integrals)
            return z.convert_to_tensor(integral)
        else:
            raise SpecificFunctionNotImplemented

    @supports(norm=False)
    def _analytic_integrate(self, limits, norm):
        if self._prod_is_same_obs_pdf:
            raise AnalyticIntegralNotImplemented(
                f"Cannot integrate analytically as PDFs have overlapping obs:"
                f" {[pdf.obs for pdf in self.pdfs]}"
            )
        integrals = []
        for pdf in self._prod_disjoint_obs_pdfs:
            limit = limits.with_obs(pdf.obs)
            try:
                integral = pdf.analytic_integrate(limits=limit, norm=norm)
            except AnalyticIntegralNotImplemented:
                raise AnalyticIntegralNotImplemented(
                    f"At least one pdf ({pdf} does not support analytic integration."
                )
            else:
                integrals.append(integral)
        integral = functools.reduce(operator.mul, integrals)
        return z.convert_to_tensor(integral)

    @supports(multiple_limits=True, norm=True)
    def _partial_integrate(self, x, limits, norm, *, options):
        if self._prod_is_same_obs_pdf:
            raise SpecificFunctionNotImplemented
        pdfs = self._prod_disjoint_obs_pdfs

        values = []
        for pdf in pdfs:
            intersection_limits = set(pdf.obs).intersection(limits.obs)
            intersection_data = set(pdf.obs).intersection(x.obs)
            if intersection_limits and not intersection_data:
                values.append(pdf.integrate(limits=limits, norm=norm, options=options))
            elif intersection_limits:  # implicitly "and intersection_data"
                values.append(
                    pdf.partial_integrate(x=x, limits=limits, options=options)
                )
            else:
                assert (
                    not intersection_limits and intersection_data
                ), "Something slipped, the logic is flawed."
                values.append(pdf.pdf(x, norm_range=norm))
        values = functools.reduce(operator.mul, values)
        return z.convert_to_tensor(values)

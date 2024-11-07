"""Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the models easier.

Their implementation is often non-trivial.
"""

#  Copyright (c) 2024 zfit

from __future__ import annotations

import functools
import operator
from collections import Counter
from collections.abc import Iterable
from typing import Literal, Optional

import pydantic.v1 as pydantic
import tensorflow as tf

import zfit.data
import zfit.z.numpy as znp

from .. import z
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData, ZfitPDF, ZfitSpace
from ..core.serialmixin import SerializableMixin
from ..core.space import supports
from ..models.basefunctor import FunctorMixin, extract_daughter_input_obs
from ..serialization import Serializer
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (
    AnalyticIntegralNotImplemented,
    NormRangeUnderdefinedError,
    SpecificFunctionNotImplemented,
)
from ..util.plotter import PDFPlotter, SumCompPlotter
from ..util.ztyping import ExtendedInputType, NormInputType
from ..z.random import counts_multinomial
from .basefunctor import FunctorPDFRepr, _preprocess_init_sum


# TODO: order of spaces if the obs is different from the wrapped pdf
class BaseFunctor(FunctorMixin, BasePDF):
    def __init__(self, pdfs, name="BaseFunctor", label=None, **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(models=self.pdfs, name=name, label=label, **kwargs)
        self._set_norm_from_daugthers()

    def _set_norm_from_daugthers(self):
        norm = super().norm
        if not norm.limits_are_set:
            norm = extract_daughter_input_obs(obs=norm, spaces=[model.space for model in self.models])
        if not norm.limits_are_set:
            msg = f"Daughter pdfs {self.pdfs} do not agree on a `norm` and/or no `norm`" "has been explicitly set."
            raise NormRangeUnderdefinedError(msg)
        self._norm = norm

    @property
    def pdfs_extended(self):
        return [pdf.is_extended for pdf in self.pdfs]


class SumPDF(BaseFunctor, SerializableMixin):  # TODO: add extended argument
    def __init__(
        self,
        pdfs: Iterable[ZfitPDF],
        fracs: ztyping.ParamTypeInput | None = None,
        obs: ztyping.ObsTypeInput = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "SumPDF",
        label: str | None = None,
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
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
               If not given, the observables of the pdfs are used if they agree.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|

        Raises
            ModelIncompatibleError: If model is incompatible.
        """
        original_init = {
            "fracs": convert_to_container(fracs),
            "extended": extended,
            "obs": obs,
        }
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
            frac_param_created,
        ) = _preprocess_init_sum(fracs, obs, pdfs)
        self._frac_param_created = frac_param_created
        self._fracs = param_fracs
        self._original_fracs = fracs_cleaned
        self._automatically_extended = False

        if extended is None:
            extended = all_extended and not fracs_cleaned
        if extended is True and all_extended and not fracs_cleaned:
            self._automatically_extended = True
            extended = sum_yields
        super().__init__(pdfs=pdfs, obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)
        self.hs3.original_init.update(original_init)
        self.plot = PDFPlotter(self, componentplotter=SumCompPlotter(self))

    @property
    def fracs(self):
        return self._fracs

    def _apply_yield(self, value: float, norm: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm=norm, log=log)

    @supports()
    def _unnormalized_pdf(self, x, params):  # NOT _pdf, as the normalization range can differ
        pdfs = self.pdfs
        fracs = params.values()
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)  # to keep the broadcasting ability
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

    @supports(norm=True, multiple_limits=True)
    def _ext_pdf(self, x, norm):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1
        if (norm and not equal_norm_ranges) or not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        pdfs = self.pdfs
        probs = [pdf.ext_pdf(x) for pdf in pdfs]
        prob = sum(probs)
        return z.convert_to_tensor(prob)

    @supports(multiple_limits=True)
    def _integrate(self, limits, norm, options):
        del norm  # not supported
        pdfs = self.pdfs
        fracs = self.fracs
        # TODO(SUM): why was this needed?
        # assert norm_range not in (None, False), "Bug, who requested an unnormalized integral?"
        integrals = [
            frac * pdf.integrate(limits=limits, options=options)  # do NOT propagate the norm_range!
            for pdf, frac in zip(pdfs, fracs)
        ]
        return znp.sum(integrals, axis=0)

    @supports(multiple_limits=True)
    def _ext_integrate(self, limits, norm, options):
        del norm  # not supported
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        pdfs = self.pdfs
        # TODO(SUM): why was this needed?
        # assert norm not in (None, False), "Bug, who requested an unnormalized integral?"
        integrals = [
            pdf.ext_integrate(limits=limits, options=options)  # do NOT propagate the norm!
            for pdf in pdfs
        ]
        return znp.sum(integrals, axis=0)

    @supports(multiple_limits=True)
    def _analytic_integrate(self, limits, norm):
        del norm  # not supported
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            integrals = [
                frac * pdf.analytic_integrate(limits=limits)  # do NOT propagate the norm!
                for pdf, frac in zip(pdfs, fracs)
            ]
        except AnalyticIntegralNotImplemented as error:
            msg = (
                f"analytic_integrate of pdf {self.name} is not implemented in this"
                f" SumPDF, as at least one sub-pdf does not implement it."
            )
            raise AnalyticIntegralNotImplemented(msg) from error

        integral = sum(integrals)
        return z.convert_to_tensor(integral)

    @supports(multiple_limits=True)
    def _partial_integrate(self, x, limits, norm, *, options):
        del norm  # not supported
        pdfs = self.pdfs
        fracs = self.fracs

        # do NOT propagate the norm!
        partial_integral = [
            pdf.partial_integrate(x=x, limits=limits, options=options) * frac for pdf, frac in zip(pdfs, fracs)
        ]
        partial_integral = sum(partial_integral)
        return z.convert_to_tensor(partial_integral)

    @supports(multiple_limits=True)
    def _partial_analytic_integrate(self, x, limits, norm, options):
        del norm, options  # not supported/ignored
        pdfs = self.pdfs
        fracs = self.fracs
        try:
            partial_integral = [
                pdf.partial_analytic_integrate(x=x, limits=limits) * frac
                # do NOT propagate the norm!
                for pdf, frac in zip(pdfs, fracs)
            ]
        except AnalyticIntegralNotImplemented as error:
            msg = (
                "partial_analytic_integrate of pdf {name} is not implemented in this"
                " SumPDF, as at least one sub-pdf does not implement it."
            )
            raise AnalyticIntegralNotImplemented(msg) from error
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
        return z.random.shuffle(sample)


class SumPDFRepr(FunctorPDFRepr):
    _implementation = SumPDF
    hs3_type: Literal["SumPDF"] = pydantic.Field("SumPDF", alias="type")
    fracs: Optional[list[Serializer.types.ParamInputTypeDiscriminated]] = None

    @pydantic.root_validator(pre=True)
    def validate_all_sumpdf(cls, values):
        # the created variable could be used, i.e. the composed autoparameter, so it should be the same
        # if cls.orm_mode(values):
        #     init = values["hs3"].original_init
        #     values = dict(values)
        #     values["fracs"] = init["fracs"]
        return values


class ProductPDF(BaseFunctor, SerializableMixin):
    def __init__(
        self,
        pdfs: list[ZfitPDF],
        obs: ztyping.ObsTypeInput = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name="ProductPDF",
    ):
        """Product of multiple PDFs in the same or different variables.

        This implementation optimizes the integration: if the PDFs are in exclusive observables, the
        integration can be factorized and the integrals can be calculated separately. This also takes into account
        that only some PDFs might have overlapping observables.

        Args:
            pdfs: List of PDFs to multiply.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
        """
        original_init = {"extended": extended, "obs": obs}

        super().__init__(pdfs=pdfs, obs=obs, name=name, extended=extended, norm=norm)
        self.hs3.original_init.update(original_init)

        # check if the pdfs have overlapping observables and separate them
        # the ones without overlapping observables can be integrated and sampled separately
        # the ones with overlapping observables need to be integrated together
        # Therefore, the overlapping ones are put into a separate ProductPDF, meaning we end up with
        # product pdfs that have either all overlapping or all disjoint observables.
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
        else:  # we need this third category, that means we cannot do any tricks as all are overlapping
            self._prod_is_same_obs_pdf = True
            self._prod_disjoint_obs_pdfs = None  # cannot add self here as it would be a circular reference

    def _unnormalized_pdf(self, x: ztyping.XType):
        probs = [pdf.pdf(x, norm=False) for pdf in self.pdfs]
        prob = functools.reduce(operator.mul, probs)
        return z.convert_to_tensor(prob)

    @supports(norm=True, multiple_limits=True)
    def _pdf(self, x, norm):
        equal_norm_ranges = len(set([pdf.norm for pdf in self.pdfs] + [norm])) == 1  # all equal
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
                integrals.append(pdf.integrate(limits=limit, norm=norm, options=options))
            integral = functools.reduce(operator.mul, integrals)
            return z.convert_to_tensor(integral)
        else:
            raise SpecificFunctionNotImplemented

    @supports(norm=False)
    def _analytic_integrate(self, limits, norm):
        if self._prod_is_same_obs_pdf:
            msg = f"Cannot integrate analytically as PDFs have overlapping obs:" f" {[pdf.obs for pdf in self.pdfs]}"
            raise AnalyticIntegralNotImplemented(msg)
        integrals = []
        for pdf in self._prod_disjoint_obs_pdfs:
            limit = limits.with_obs(pdf.obs)
            try:
                integral = pdf.analytic_integrate(limits=limit, norm=norm)
            except AnalyticIntegralNotImplemented:
                msg = f"At least one pdf ({pdf} does not support analytic integration."
                raise AnalyticIntegralNotImplemented(msg) from None
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
                values.append(pdf.partial_integrate(x=x, limits=limits, options=options))
            else:
                has_data_but_no_limits = (not intersection_limits) and intersection_data
                assert has_data_but_no_limits, "Something slipped, the logic is flawed."
                values.append(pdf.pdf(x, norm=norm))
        values = functools.reduce(operator.mul, values)
        return z.convert_to_tensor(values)

    @supports(multiple_limits=True)
    def _sample(self, n, limits: ZfitSpace):
        if self._prod_is_same_obs_pdf:
            raise SpecificFunctionNotImplemented
        pdfs = self._prod_disjoint_obs_pdfs
        samples = [pdf.sample(n=n, limits=limits.with_obs(pdf.obs)) for pdf in pdfs]
        # samples = [sample.value() if isinstance(sample, ZfitData) else sample for sample in samples]
        for sample in samples:
            assert isinstance(sample, ZfitData), "Sample must be a ZfitData"
        return zfit.data.concat(samples, axis=1).value()


class ProductPDFRepr(FunctorPDFRepr):
    _implementation = ProductPDF
    hs3_type: Literal["ProductPDF"] = pydantic.Field("ProductPDF", alias="type")

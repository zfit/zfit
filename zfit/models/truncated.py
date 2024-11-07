#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import Iterable, Literal, Union

import numpy as np
import pydantic.v1 as pydantic
import tensorflow as tf

import zfit.z.numpy as znp

from .. import z
from ..core.interfaces import ZfitPDF, ZfitSpace
from ..core.serialmixin import SerializableMixin
from ..core.space import Space, convert_to_space, supports
from ..serialization import Serializer, SpaceRepr  # noqa: F401
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import AnalyticIntegralNotImplemented, SpecificFunctionNotImplemented
from .basefunctor import FunctorPDFRepr
from .functor import BaseFunctor


def check_limits(limits: Union[ZfitSpace, list[ZfitSpace]], obs=None):
    """Check if the limits are valid Spaces and return an iterable."""
    limits = convert_to_container(limits, container=tuple)
    obs = obs.obs if obs is not None else None

    newlimits = []
    if limits is not None:
        # check if it's a list of exactly two limits given as numbers or arrays
        if (
            obs is not None
            and (
                np.shape(limits) == (2, len(obs))
                or (
                    len(limits) == 2
                    and np.atleast_1d(limits[0]).shape == (len(obs),)
                    and np.atleast_1d(limits[1]).shape == (len(obs),)
                )
            )
            and not (isinstance(limits[0], ZfitSpace) or isinstance(limits[1], ZfitSpace))
        ):
            limits = [Space(obs=obs, limits=limits)]
        notspace = []
        for limit in limits:
            if not isinstance(limit, ZfitSpace):
                if obs is None:
                    notspace.append(limit)
                else:
                    limit = Space(obs=obs, limits=limit)
            newlimits.append(limit)
        limits = newlimits
        if notspace:
            msg = f"limits {notspace} are not of type ZfitSpace and obs could not be automatically determined."
            raise TypeError(msg)
    limits_sorted = tuple(sorted(limits, key=lambda limit: limit.v1.lower))
    return check_overlap(limits_sorted)


def check_overlap(limits):
    if len(limits) == 1:
        return limits
    for i, limit1 in enumerate(limits[:-1]):
        for limit2 in limits[i + 1 :]:
            if limit1.v1.upper > limit2.v1.lower:
                msg = f"Limit {limit1} overlaps with {limit2} in TruncatedPDF."
                raise ValueError(msg)
    return limits


# TODO: implement smart limits
# def create_subset_limits(limits, constraints):
#     limits = check_limits(limits)
#     newlimits = []
#     obs = limits[0].obs
#     axes = obs.axis
#     for limit in limits:
#         if not limit.ndims == 1:
#             raise ValueError(f"Limit {limit} is not 1-dimensional.")
#         newlower, newupper = None, None
#         for constr in constraints:
#             if constr.v1.lower <= limit.v1.lower <= constr.v1.upper:
#                 assert newlower is None, "Multiple limits overlap with the same limit, should have been caught before. All limits: {limits}"
#                 newlower = max(limit.v1.lower, constr.v1.lower)
#             if newlower and constr.v1.lower <= limit.v1.upper <= constr.v1.upper:
#                 assert newupper is None, "Multiple limits overlap with the same limit, should have been caught before. All limits: {limits}"
#                 newupper = min(limit.v1.upper, constr.v1.upper)
#         if newlower is not None and newupper is not None:
#             newlimits.append(Space(obs=obs, lower=newlower, upper=newupper, axes=axes))
#             break


class TruncatedPDF(BaseFunctor, SerializableMixin):
    def __init__(
        self,
        pdf: ZfitPDF,
        limits: ZfitSpace | Iterable[ZfitSpace],
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ztyping.ExtendedInputType = None,
        norm: ztyping.NormRangeTypeInput = None,
        name: str | None = None,
        label: str | None = None,
    ):
        """Truncated PDF in one or multiple ranges.

        The PDF is truncated to the given limits, i.e. the PDF is only evaluated within the given limits
        and outside the limits, the PDF is zero. The limits can be given as a single or as a list of limits.
        This solves two problems: first, the PDF is only evaluated where it is defined and zero outside,
        which enables PDFs that are not defined outside the limits to be used (Poisson, LogNormal, Chi2,...).
        Second, the list of limits can be used to effectively create a multilimit PDF, that means, a PDF that
        is defined in multiple ranges and zero outside these ranges, also in between.

        Historically, this was implemented in zfit through MultipleLimits, which is now deprecated and replaced
        by this class.

        Args:
            pdf: The PDF to be truncated.
            limits: The limits to truncate the PDF. Can be a single limit or multiple limits.
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
               If None, the PDF will be extended if the original PDF is extended.
               If ``True`` and the original PDF is extended, the yield will be scaled to the
               fraction of the total integral that is within the limits.
               Therefore, the overall yield is comparable, i.e. the pdfs can be plotted
               "on top of each other".
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        original_init = {"extended": extended, "obs": obs}
        if name is None:
            name = "TruncatedPDF"
        obs = pdf.space if obs is None else convert_to_space(obs)
        self._limits = check_limits(limits, obs=obs)
        if extended is None:
            extended = pdf.is_extended
        if extended is True and pdf.is_extended:
            paramname = "wrapped_yield"

            def scaled_yield(params):
                base_norm = pdf.integrate(limits=obs, norm=False)
                piecewise_norms = znp.asarray([pdf.integrate(limits=limit, norm=False) for limit in self._limits])
                relative_scale = znp.sum(piecewise_norms / base_norm)
                return (params[paramname] * relative_scale,)

            import zfit

            params_deps = {p.name: p for p in pdf.get_params(floating=None)}
            toreplace = paramname
            while toreplace in params_deps:
                newtoreplace = f"{toreplace}_x"
                params_deps[newtoreplace] = params_deps.pop(toreplace)
                toreplace = newtoreplace

            pdfyield = pdf.get_yield()
            params_deps[paramname] = pdfyield
            if pdfyield.name in params_deps:
                params_deps.pop(pdfyield.name)

            extended = zfit.param.ComposedParameter(
                name=f"AUTO_yield{zfit.core.parameter.get_auto_number()!s}_{name}",
                func=scaled_yield,
                params=params_deps,
            )
        super().__init__(obs=obs, name=name, extended=extended, norm=norm, pdfs=pdf, label=label)
        if self.obs != pdf.obs:
            msg = f"The space of the TruncatedPDF ({self.obs}) must be the same as the PDF ({pdf.space})."
            raise ValueError(msg)
        self.hs3.original_init.update(original_init)

    @property
    def limits(self):
        return self._limits

    def _unnormalized_pdf(self, x):
        # the implementation only feeds the pdf with data that is inside the limits that we want to evaluate
        # this maybe has a speedup (although limited as we're using dynamic shapes -> needs to change for JAX)
        # but most importantly avoids any issue if we take the gradient of a pdf that is not defined outside
        # the limits, because this can yield NaNs using a naive multiplication with a mask
        from zfit import Data

        xarray = znp.asarray(x.value())
        inside_arrays = [limit.inside(x) for limit in self._limits]
        inside = znp.any(inside_arrays, axis=0)
        indices = znp.transpose(znp.where(inside))  # is a list? but transpose gives perfect shape for indices
        data = Data.from_tensor(tensor=xarray[inside], obs=self.obs)
        prob = self.pdfs[0].pdf(data, norm=False)
        return tf.scatter_nd(indices, prob, tf.shape(xarray, out_type=np.int64)[:1])  # only nevents

    @supports()
    def _integrate(self, limits, norm, options=None):
        del norm  # not used here
        # cannot equal, as possibly jitted
        from zfit import run

        if (
            not run.executing_eagerly() or limits != self.space
        ):  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        # limits = create_subset_limits(limits, self.limits)  # TODO: be smart about limits, we would not need to throw the SpecificFunctionNotImplemented
        integrals = [self.pdfs[0].integrate(limits=limit, norm=False, options=options) for limit in limits]
        return znp.sum(integrals, axis=0)

    @supports()
    def _analytic_integrate(self, limits, norm):
        del norm  # not used here
        # cannot equal, as possibly jitted
        from zfit import run

        if (
            not run.executing_eagerly() or limits != self.space
        ):  # we could also do it, but would need to check each limit
            raise AnalyticIntegralNotImplemented
        limits = convert_to_container(
            self.limits
        )  # if it's the overarching limits, we can just use our own ones, the real ones
        # limits = create_subset_limits(limits, self.limits)  # TODO: be smart about limits, we would not need to throw the SpecificFunctionNotImplemented
        integrals = [self.pdfs[0].analytic_integrate(limits=limit, norm=False) for limit in limits]
        return znp.sum(integrals, axis=0)

    # TODO: we could make sampling more efficient by only sampling the relevant ranges, however, that would
    # mean we need to check if the limits of the pdf are within the limits given
    @supports()
    def _sample(self, n, limits):
        pdf = self.pdfs[0]

        # TODO: cannot compare, as possibly jitted
        from zfit import run

        if (
            not run.executing_eagerly() or limits != self.space
        ):  # we could also do it, but would need to check each limit
            raise SpecificFunctionNotImplemented
        limits = self.limits
        # should be `self.integrate`, but as we do it numerically currently, more efficient to use pdf
        if len(limits) > 1:
            integrals = znp.concatenate([pdf.integrate(limits=limit, norm=False) for limit in limits])
            fracs = integrals / znp.sum(integrals, axis=0)  # norm
            fracs.set_shape([len(limits)])
            counts = tf.unstack(z.random.counts_multinomial(n, probs=fracs), axis=0)
        else:
            counts = [n]
        samples = [self.pdfs[0].sample(count, limits=limit).value() for count, limit in zip(counts, limits)]
        return znp.concatenate(samples, axis=0)


class TruncatedPDFRepr(FunctorPDFRepr):
    _implementation = TruncatedPDF
    hs3_type: Literal["TruncatedPDF"] = pydantic.Field("TruncatedPDF", alias="type")
    limits: list[SpaceRepr]

    def _to_orm(self, init):
        init["pdf"] = init.pop("pdfs")[0]
        return super()._to_orm(init)

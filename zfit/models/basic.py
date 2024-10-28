"""Basic PDFs are provided here.

Gauss, exponential... that can be used together with Functors to build larger models.
"""

#  Copyright (c) 2024 zfit

from __future__ import annotations

#  Copyright (c) 2023 zfit
import contextlib
from typing import Literal

import numpy as np
import tensorflow as tf
from pydantic.v1 import Field

import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.serialmixin import SerializableMixin
from ..core.space import ANY_LOWER, ANY_UPPER, Space, supports
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..util import ztyping
from ..util.exception import BreakingAPIChangeError
from ..util.warnings import warn_advanced_feature
from ..util.ztyping import ExtendedInputType, NormInputType


class Exponential(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        lam=None,
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Exponential",
        lambda_=None,
        label: str | None = None,
    ):
        """Exponential function exp(lambda * x).

        The function is normalized over a finite range and therefore a pdf. So the PDF is precisely
        defined as :math:`\\frac{ e^{\\lambda \\cdot x}}{ \\int_{lower}^{upper} e^{\\lambda \\cdot x} dx}`

        Args:
            lam: Lambda parameter of the exponential.
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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        if lambda_ is not None:
            if lam is None:
                lam = lambda_
            else:
                msg = "The 'lambda' parameter has been renamed from 'lambda_' to 'lam'."
                raise BreakingAPIChangeError(msg)
        params = {"lambda": lam, "lam": lam}
        super().__init__(obs, name=name, params=params, extended=extended, norm=norm, label=label)

        self._calc_numerics_data_shift = lambda: z.constant(0.0)

        if not self.space.has_limits:
            warn_advanced_feature(
                "Exponential pdf relies on a shift of the input towards 0 to keep the numerical "
                f"stability high. The space {self.space} does not have limits set and no shift"
                f" will occure. To set it manually, set _numerics_data_shift to the expected"
                f" average values given to this function _in case you want things to be set_."
                f"If this sounds unfamiliar, regard this as an error and use a normalization range.",
                identifier="exp_shift",
            )
        self._set_numerics_data_shift(self.space)

    @supports()
    def _unnormalized_pdf(self, x, params):
        lambda_ = params["lambda"]
        x = x.unstack_x()
        probs = znp.exp(lambda_ * (self._shift_x(x)))
        tf.debugging.assert_all_finite(
            probs,
            f"Exponential PDF {self} has non valid values. This is likely caused"
            f" by numerical problems: if the exponential is too steep, this will"
            f" yield NaNs or infs. Make sure that your lambda is small enough and/or"
            f" the initial space is in the same"
            f" region as your data (and norm, if explicitly set differently)."
            f" If this issue still persists, please oben an issue on Github:"
            f" https://github.com/zfit/zfit",
        )
        return probs  # Don't use exp! will overflow.

    def _shift_x(self, x):
        return x - self._calc_numerics_data_shift()

    @contextlib.contextmanager
    def _set_numerics_data_shift(self, limits):
        if limits:

            def calc_numerics_data_shift():
                lower, upper = [], []
                for limit in limits:
                    low, up = limit.rect_limits
                    lower.append(z.convert_to_tensor(low[:, 0]))
                    upper.append(z.convert_to_tensor(up[:, 0]))
                lower = z.convert_to_tensor(lower)
                upper = z.convert_to_tensor(upper)
                lower_val = znp.min(lower, axis=0)
                upper_val = znp.max(upper, axis=0)

                return (upper_val + lower_val) / 2

            old_value = self._calc_numerics_data_shift

            self._calc_numerics_data_shift = calc_numerics_data_shift
            yield
            self._calc_numerics_data_shift = old_value
        else:
            yield

    # All hooks are needed to set the right shift when "entering" the pdf. The norm range is taken where both are
    # available. No special need needs to be taken for sampling (it samples from the correct region, the limits, and
    # uses the predictions by the `unnormalized_prob` -> that is shifted correctly
    def _single_hook_integrate(self, limits, norm, x, options):
        with self._set_numerics_data_shift(norm):
            return super()._single_hook_integrate(limits, norm, x=x, options=options)

    def _single_hook_analytic_integrate(self, limits, norm):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_analytic_integrate(limits, norm)

    def _single_hook_numeric_integrate(self, limits, norm, options):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_numeric_integrate(limits, norm, options)

    def _single_hook_partial_integrate(self, x, limits, norm, *, options):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_partial_integrate(x, limits, norm, options=options)

    def _single_hook_partial_analytic_integrate(self, x, limits, norm):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_partial_analytic_integrate(x, limits, norm)

    def _single_hook_partial_numeric_integrate(self, x, limits, norm):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_partial_numeric_integrate(x, limits, norm)

    # def _single_hook_normalization(self, limits):
    #     with self._set_numerics_data_shift(limits=limits):
    #         return super()._single_hook_normalization(limits)

    #
    # # TODO: remove component_norm_range? But needed for integral?
    # def _single_hook_unnormalized_pdf(self, x, name):
    #     if component_norm_range.limits_are_false:
    #         component_norm_range = self.space
    #     if component_norm_range.limits_are_set:
    #         with self._set_numerics_data_shift(limits=component_norm_range):
    #             return super()._single_hook_unnormalized_pdf(x, name)
    #     else:
    #         return super()._single_hook_unnormalized_pdf(x, name)
    #
    def _single_hook_pdf(self, x, norm):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_pdf(x, norm)

    def _single_hook_log_pdf(self, x, norm):
        with self._set_numerics_data_shift(limits=norm):
            return super()._single_hook_log_pdf(x, norm)

    def _single_hook_sample(self, n, limits, x=None):
        with self._set_numerics_data_shift(limits=limits):
            return super()._single_hook_sample(n, limits, x)


def _exp_integral_from_any_to_any(limits, params, model):
    lambda_ = params["lambda"]
    lower, upper = limits.rect_limits  # TODO: change to v1 limits?
    # if any(np.isinf([lower, upper])):
    #     raise AnalyticIntegralNotImplemented

    integral = _exp_integral_func_shifting(lambd=lambda_, lower=lower, upper=upper, model=model)
    return integral[0]


def _exp_integral_func_shifting(lambd, lower, upper, model):
    def raw_integral(x):
        return z.exp(lambd * (model._shift_x(x))) / lambd  # needed due to overflow in exp otherwise

    lower = z.convert_to_tensor(lower)
    lower_int = raw_integral(x=lower)
    upper = z.convert_to_tensor(upper)
    upper_int = raw_integral(x=upper)
    return upper_int - lower_int


def exp_icdf(x, params, model):
    lambd = params["lambda"]
    x = z.unstack_x(x)
    x = model._shift_x(x)
    return znp.log(lambd * x) / lambd


# Exponential.register_inverse_analytic_integral(exp_icdf)  # TODO: register icdf for exponential
# TODO: cleanup, make cdf registrable _and_ inverse integral, but real

limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Exponential.register_analytic_integral(func=_exp_integral_from_any_to_any, limits=limits)


class ExponentialPDFRepr(BasePDFRepr):
    _implementation = Exponential
    hs3_type: Literal["Exponential"] = Field("Exponential", alias="type")
    x: SpaceRepr
    lam: Serializer.types.ParamTypeDiscriminated


class Voigt(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        m: ztyping.ObsTypeInput,
        sigma: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Voigt",
        label: str | None = None,
    ):
        """Voigt profile.

        The Voigt profile is a convolution of a 1-D Gaussian distribution with standard deviation sigma and a 1-D Cauchy distribution with half-width at half-maximum gamma.

        The (unnormalized) Voigt profile shape is defined as

        .. math::

            V(x; \\sigma, \\gamma) = Re[w(z)]

        where :math:`z = \\frac{x + i\\gamma}{\\sqrt{2}\\sigma}` and :math:`w(z)` is the Faddeeva function

        with the normalization over [-inf, inf] of

        .. math::

            \\frac{1}{\\sqrt{2\\pi\\sigma^2} }

        The normalization changes for different normalization ranges.

        :math:`m` is the mean, :math:`\\sigma` is the standard deviation of the Gaussian part and :math:`\\gamma` is the width of the Cauchy part.


        Args:
            m: Mean of the Voigt profile.
            sigma: Standard deviation or spread of the Gaussian part.
            gamma: Width of the shape of the Cauchy part.
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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        params = {"m": m, "sigma": sigma, "gamma": gamma}
        super().__init__(obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports()
    def _unnormalized_pdf(self, x, params):
        m = params["m"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        x = z.unstack_x(x)
        x = znp.asarray(znp.atleast_1d(x) - m, dtype=znp.complex128)
        gamma_complex = znp.asarray(gamma, dtype=znp.complex128)
        sigma_complex = znp.asarray(sigma, dtype=znp.complex128)
        complex_z = (x + 1j * gamma_complex) / (np.sqrt(2) * sigma_complex)
        return znp.real(znp.faddeeva_humlicek(complex_z)) / (sigma * np.sqrt(2 * np.pi))


def _voigt_integral_from_inf_to_inf(limits, params, model):
    del model, limits  # unused, fixed limits
    params["m"]
    sigma = params["sigma"]
    params["gamma"]
    return sigma * np.sqrt(2 * np.pi)


# do NOT uncomment, this can lead to deadlocks. No joke: https://github.com/tensorflow/tensorflow/issues/66115
# limits = Space(axes=0, limits=(-znp.inf, znp.inf))


# todo: this only works if executing eagerly, which fails for at least the binned PDFs
# possible solution comes with the `space.static` concept (?) and a StaticLimitsNotAvailable error,
# falling back to whatever is available
#  Voigt.register_analytic_integral(func=_voigt_integral_from_inf_to_inf, limits=limits)
class VoigtPDFRepr(BasePDFRepr):
    _implementation = Voigt
    hs3_type: Literal["Voigt"] = Field("Voigt", alias="type")
    x: SpaceRepr
    m: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    gamma: Serializer.types.ParamTypeDiscriminated

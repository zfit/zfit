#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import Literal

import numpy as np
import pydantic.v1 as pydantic
import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z
from ..core.basepdf import BasePDF
from ..core.serialmixin import SerializableMixin
from ..core.space import ANY_LOWER, ANY_UPPER, Space, supports
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..util import ztyping
from ..util.ztyping import ExtendedInputType, NormInputType


def _powerlaw(x, a, k):
    return a * znp.power(x, k)


@z.function(wraps="tensor", keepalive=True)
def crystalball_func(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = znp.abs(alpha)
    a = znp.power((n / abs_alpha), n) * znp.exp(-0.5 * znp.square(alpha))
    b = (n / abs_alpha) - abs_alpha
    cond = tf.less(t, -abs_alpha)
    func = z.safe_where(
        cond,
        lambda t: _powerlaw(b - t, a, -n),
        lambda t: znp.exp(-0.5 * znp.square(t)),
        values=t,
        value_safer=lambda t: znp.ones_like(t) * (b - 2),
    )
    return znp.maximum(func, znp.zeros_like(func))


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def double_crystalball_func(x, mu, sigma, alphal, nl, alphar, nr):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        crystalball_func(x, mu, sigma, alphal, nl),
        crystalball_func(x, mu, sigma, -alphar, nr),
    )


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def generalized_crystalball_func(x, mu, sigmal, alphal, nl, sigmar, alphar, nr):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        crystalball_func(x, mu, sigmal, alphal, nl),
        crystalball_func(x, mu, sigmar, -alphar, nr),
    )


# created with the help of TensorFlow autograph used on python code converted from ShapeCB of RooFit
def crystalball_integral(limits, params, model):
    del model
    mu = params["mu"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    n = params["n"]

    lower, upper = limits._rect_limits_tf

    return crystalball_integral_func(mu, sigma, alpha, n, lower, upper)


@z.function(wraps="tensor", keepalive=True)
def crystalball_integral_func(mu, sigma, alpha, n, lower, upper):
    sqrt_pi_over_two = np.sqrt(np.pi / 2)
    sqrt2 = np.sqrt(2)

    use_log = tf.less(znp.abs(n - 1.0), 1e-05)
    abs_sigma = znp.abs(sigma)
    abs_alpha = znp.abs(alpha)
    tmin = (lower - mu) / abs_sigma
    tmax = (upper - mu) / abs_sigma

    alpha_negative = tf.less(alpha, 0)
    # do not move on two lines, logic will fail...
    tmax, tmin = (
        znp.where(alpha_negative, -tmin, tmax),
        znp.where(alpha_negative, -tmax, tmin),
    )

    if_true_4 = abs_sigma * sqrt_pi_over_two * (tf.math.erf(tmax / sqrt2) - tf.math.erf(tmin / sqrt2))

    a = znp.power(n / abs_alpha, n) * znp.exp(-0.5 * tf.square(abs_alpha))
    b = n / abs_alpha - abs_alpha

    # gradients from tf.where can be NaN if the non-selected branch is NaN
    # https://github.com/tensorflow/tensorflow/issues/42889
    # solution is to provide save values for the non-selected branch to never make them become NaNs
    b_tmin = b - tmin
    safe_b_tmin_ones = znp.where(b_tmin > 0, b_tmin, znp.ones_like(b_tmin))
    b_tmax = b - tmax
    safe_b_tmax_ones = znp.where(b_tmax > 0, b_tmax, znp.ones_like(b_tmax))

    if_true_1 = a * abs_sigma * (znp.log(safe_b_tmin_ones) - znp.log(safe_b_tmax_ones))

    if_false_1 = (
        a
        * abs_sigma
        / (1.0 - n)
        * (1.0 / znp.power(safe_b_tmin_ones, n - 1.0) - 1.0 / znp.power(safe_b_tmax_ones, n - 1.0))
    )

    if_true_3 = tf.where(use_log, if_true_1, if_false_1)

    if_true_2 = a * abs_sigma * (znp.log(safe_b_tmin_ones) - znp.log(n / abs_alpha))
    if_false_2 = (
        a
        * abs_sigma
        / (1.0 - n)
        * (1.0 / znp.power(safe_b_tmin_ones, n - 1.0) - 1.0 / znp.power(n / abs_alpha, n - 1.0))
    )
    term1 = tf.where(use_log, if_true_2, if_false_2)
    term2 = abs_sigma * sqrt_pi_over_two * (tf.math.erf(tmax / sqrt2) - tf.math.erf(-abs_alpha / sqrt2))
    if_false_3 = term1 + term2

    if_false_4 = tf.where(tf.less_equal(tmax, -abs_alpha), if_true_3, if_false_3)

    # if_false_4()
    result = tf.where(tf.greater_equal(tmin, -abs_alpha), if_true_4, if_false_4)
    if result.shape.rank != 0:
        result = tf.gather(result, 0, axis=-1)  # remove last dim, should vanish
    return result


def double_crystalball_mu_integral(limits, params, model):
    del model
    mu = params["mu"]
    sigma = params["sigma"]
    alphal = params["alphal"]
    nl = params["nl"]
    alphar = params["alphar"]
    nr = params["nr"]

    lower, upper = limits._rect_limits_tf
    lower = lower[:, 0]
    upper = upper[:, 0]

    return double_crystalball_mu_integral_func(
        mu=mu,
        sigma=sigma,
        alphal=alphal,
        nl=nl,
        alphar=alphar,
        nr=nr,
        lower=lower,
        upper=upper,
    )


@z.function(wraps="tensor")  # TODO: this errors, fro whatever reason?
def double_crystalball_mu_integral_func(mu, sigma, alphal, nl, alphar, nr, lower, upper):
    # mu_broadcast =
    upper_of_lowerint = znp.minimum(mu, upper)
    integral_left = crystalball_integral_func(
        mu=mu, sigma=sigma, alpha=alphal, n=nl, lower=lower, upper=upper_of_lowerint
    )
    left = tf.where(tf.less(mu, lower), znp.zeros_like(integral_left), integral_left)

    lower_of_upperint = znp.maximum(mu, lower)
    integral_right = crystalball_integral_func(
        mu=mu, sigma=sigma, alpha=-alphar, n=nr, lower=lower_of_upperint, upper=upper
    )
    right = tf.where(tf.greater(mu, upper), znp.zeros_like(integral_right), integral_right)

    return left + right


def generalized_crystalball_mu_integral(limits, params, model):
    del model
    mu = params["mu"]
    sigmal = params["sigmal"]
    alphal = params["alphal"]
    nl = params["nl"]
    sigmar = params["sigmar"]
    alphar = params["alphar"]
    nr = params["nr"]

    lower, upper = limits._rect_limits_tf
    lower = lower[:, 0]
    upper = upper[:, 0]

    return generalized_crystalball_mu_integral_func(
        mu=mu,
        sigmal=sigmal,
        alphal=alphal,
        nl=nl,
        sigmar=sigmar,
        alphar=alphar,
        nr=nr,
        lower=lower,
        upper=upper,
    )


@z.function(wraps="tensor")
def generalized_crystalball_mu_integral_func(mu, sigmal, alphal, nl, sigmar, alphar, nr, lower, upper):
    # mu_broadcast =
    upper_of_lowerint = znp.minimum(mu, upper)
    integral_left = crystalball_integral_func(
        mu=mu, sigma=sigmal, alpha=alphal, n=nl, lower=lower, upper=upper_of_lowerint
    )
    left = tf.where(tf.less(mu, lower), znp.zeros_like(integral_left), integral_left)

    lower_of_upperint = znp.maximum(mu, lower)
    integral_right = crystalball_integral_func(
        mu=mu, sigma=sigmar, alpha=-alphar, n=nr, lower=lower_of_upperint, upper=upper
    )
    right = tf.where(tf.greater(mu, upper), znp.zeros_like(integral_right), integral_right)

    return left + right


class CrystalBall(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alpha: ztyping.ParamTypeInput,
        n: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "CrystalBall",
        label: str | None = None,
    ):
        """Crystal Ball shaped PDF. A combination of a Gaussian with a powerlaw tail.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha, n) =  \\begin{cases} \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & \\mbox{for }\\frac{x - \\mu}{\\sigma} \\geqslant -\\alpha \\newline
            A \\cdot (B - \\frac{x - \\mu}{\\sigma})^{-n}, & \\mbox{for }\\frac{x - \\mu}{\\sigma}
             < -\\alpha \\end{cases}

        with

        .. math::
            A = \\left(\\frac{n}{\\left| \\alpha \\right|}\\right)^n \\cdot
            \\exp\\left(- \\frac {\\left|\\alpha \\right|^2}{2}\\right)

            B = \\frac{n}{\\left| \\alpha \\right|}  - \\left| \\alpha \\right|

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alpha: parameter where to switch from a gaussian to the powertail
            n: Exponent of the powertail
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

        .. _CBShape: https://en.wikipedia.org/wiki/Crystal_Ball_function
        """
        params = {"mu": mu, "sigma": sigma, "alpha": alpha, "n": n}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        del norm
        mu = params["mu"]
        sigma = params["sigma"]
        alpha = params["alpha"]
        n = params["n"]
        x = z.unstack_x(x)
        return crystalball_func(x=x, mu=mu, sigma=sigma, alpha=alpha, n=n)


class CrystalBallPDFRepr(BasePDFRepr):
    _implementation = CrystalBall
    hs3_type: Literal["CrystalBall"] = pydantic.Field("CrystalBall", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alpha: Serializer.types.ParamTypeDiscriminated
    n: Serializer.types.ParamTypeDiscriminated


crystalball_integral_limits = Space(axes=(0,), lower=ANY_LOWER, upper=ANY_UPPER)

CrystalBall.register_analytic_integral(func=crystalball_integral, limits=crystalball_integral_limits)


class DoubleCB(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "DoubleCB",
        label: str | None = None,
    ):
        """Double-sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a frac) on each side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha_{L}, n_{L}, \\alpha_{R}, n_{R}) =  \\begin{cases}
            A_{L} \\cdot (B_{L} - \\frac{x - \\mu}{\\sigma})^{-n_{L}},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma} < -\\alpha_{L} \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & \\mbox{for }-\\alpha_{L} \\leqslant \\frac{x - \\mu}{\\sigma} \\leqslant \\alpha_{R} \\newline
            A_{R} \\cdot (B_{R} + \\frac{x - \\mu}{\\sigma})^{-n_{R}},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma} > \\alpha_{R}
            \\end{cases}

        with

        .. math::
            A_{L/R} = \\left(\\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}\\right)^n_{L/R} \\cdot
            \\exp\\left(- \\frac {\\left|\\alpha_{L/R} \\right|^2}{2}\\right)

            B_{L/R} = \\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}  - \\left| \\alpha_{L/R} \\right|

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alphal: parameter where to switch from a gaussian to the powertail on the left
                side
            nl: Exponent of the powertail on the left side
            alphar: parameter where to switch from a gaussian to the powertail on the right
                side
            nr: Exponent of the powertail on the right side
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
        params = {
            "mu": mu,
            "sigma": sigma,
            "alphal": alphal,
            "nl": nl,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm cannot be a space"
        mu = params["mu"]
        sigma = params["sigma"]
        alphal = params["alphal"]
        nl = params["nl"]
        alphar = params["alphar"]
        nr = params["nr"]
        x = x[0]
        return double_crystalball_func(
            x=x,
            mu=mu,
            sigma=sigma,
            alphal=alphal,
            nl=nl,
            alphar=alphar,
            nr=nr,
        )


class DoubleCBPDFRepr(BasePDFRepr):
    _implementation = DoubleCB
    hs3_type: Literal["DoubleCB"] = pydantic.Field("DoubleCB", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    nl: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated
    nr: Serializer.types.ParamTypeDiscriminated


DoubleCB.register_analytic_integral(func=double_crystalball_mu_integral, limits=crystalball_integral_limits)


class GeneralizedCB(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "GeneralizedCB",
        label: str | None = None,
    ):
        """Generalized asymmetric double-sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a
        frac) and a different **sigma** on each side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma_{L}, \\alpha_{L}, n_{L}, \\sigma_{R}, \\alpha_{R}, n_{R}) =  \\begin{cases}
            A_{L} \\cdot (B_{L} - \\frac{x - \\mu}{\\sigma_{L}})^{-n_{L}},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma_{L}} < -\\alpha_{L} \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma_{L}^2}),
            & \\mbox{for }-\\alpha_{L} \\leqslant \\frac{x - \\mu}{\\sigma_{L}} \\leqslant 0 \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma_{R}^2}),
            & \\mbox{for }0 \\leqslant \\frac{x - \\mu}{\\sigma_{R}} \\leqslant \\alpha_{R} \\newline
            A_{R} \\cdot (B_{R} + \\frac{x - \\mu}{\\sigma_{R}})^{-n_{R}},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma_{R}} > \\alpha_{R}
            \\end{cases}

        with

        .. math::
            A_{L/R} = \\left(\\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}\\right)^n_{L/R} \\cdot
            \\exp\\left(- \\frac {\\left|\\alpha_{L/R} \\right|^2}{2}\\right)

            B_{L/R} = \\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}  - \\left| \\alpha_{L/R} \\right|

        Args:
            mu: The mean of the gaussian
            sigmal: Standard deviation of the gaussian on the left side
            alphal: parameter where to switch from a gaussian to the powertail on the left
                side
            nl: Exponent of the powertail on the left side
            sigmar: Standard deviation of the gaussian on the right side
            alphar: parameter where to switch from a gaussian to the powertail on the right
                side
            nr: Exponent of the powertail on the right side
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
        params = {
            "mu": mu,
            "sigmal": sigmal,
            "alphal": alphal,
            "nl": nl,
            "sigmar": sigmar,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigmal = params["sigmal"]
        alphal = params["alphal"]
        sigmar = params["sigmar"]
        nl = params["nl"]
        alphar = params["alphar"]
        nr = params["nr"]
        x = x[0]
        return generalized_crystalball_func(
            x=x,
            mu=mu,
            sigmal=sigmal,
            alphal=alphal,
            sigmar=sigmar,
            nl=nl,
            alphar=alphar,
            nr=nr,
        )


class GeneralizedCBPDFRepr(BasePDFRepr):
    _implementation = GeneralizedCB
    hs3_type: Literal["GeneralizedCB"] = pydantic.Field("GeneralizedCB", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigmal: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    sigmar: Serializer.types.ParamTypeDiscriminated
    nl: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated
    nr: Serializer.types.ParamTypeDiscriminated


GeneralizedCB.register_analytic_integral(func=generalized_crystalball_mu_integral, limits=crystalball_integral_limits)


@z.function(wraps="tensor", keepalive=True)
def gaussexptail_func(x, mu, sigma, alpha):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = znp.abs(alpha)
    cond = tf.less(t, -abs_alpha)
    func = z.safe_where(
        cond,
        lambda t: 0.5 * znp.square(abs_alpha) + abs_alpha * t,
        lambda t: -0.5 * znp.square(t),
        values=t,
        value_safer=lambda t: znp.ones_like(t) * abs_alpha,
    )
    return znp.exp(func)


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def generalized_gaussexptail_func(x, mu, sigmal, alphal, sigmar, alphar):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        gaussexptail_func(x, mu, sigmal, alphal),
        gaussexptail_func(x, mu, sigmar, -alphar),
    )


def gaussexptail_integral(limits, params, model):
    del model
    mu = params["mu"]
    sigma = params["sigma"]
    alpha = params["alpha"]

    lower, upper = limits._rect_limits_tf

    return gaussexptail_integral_func(mu, sigma, alpha, lower, upper)


@z.function(wraps="tensor", keepalive=True)
def gaussexptail_integral_func(mu, sigma, alpha, lower, upper):
    sqrt_pi_over_two = np.sqrt(np.pi / 2)
    sqrt2 = np.sqrt(2)

    abs_sigma = znp.abs(sigma)
    abs_alpha = znp.abs(alpha)
    tmin = (lower - mu) / abs_sigma
    tmax = (upper - mu) / abs_sigma

    alpha_negative = tf.less(alpha, 0)
    # do not move on two lines, logic will fail...
    tmax, tmin = (
        znp.where(alpha_negative, -tmin, tmax),
        znp.where(alpha_negative, -tmax, tmin),
    )

    gauss_tmin_tmax_integral = abs_sigma * sqrt_pi_over_two * (tf.math.erf(tmax / sqrt2) - tf.math.erf(tmin / sqrt2))
    exp_tmin_tmax_integral = (
        abs_sigma
        / abs_alpha
        * znp.exp(0.5 * znp.square(abs_alpha))
        * (znp.exp(abs_alpha * tmax) - znp.exp(abs_alpha * tmin))
    )
    gauss_minus_abs_alpha_tmax_integral = (
        abs_sigma * sqrt_pi_over_two * (tf.math.erf(tmax / sqrt2) - tf.math.erf(-abs_alpha / sqrt2))
    )
    exp_tmin_minus_abs_alpha_integral = (
        abs_sigma
        / abs_alpha
        * znp.exp(0.5 * znp.square(abs_alpha))
        * (znp.exp(-znp.square(abs_alpha)) - znp.exp(abs_alpha * tmin))
    )
    integral_sum = exp_tmin_minus_abs_alpha_integral + gauss_minus_abs_alpha_tmax_integral

    conditional_integral = tf.where(tf.less_equal(tmax, -abs_alpha), exp_tmin_tmax_integral, integral_sum)
    result = tf.where(tf.greater_equal(tmin, -abs_alpha), gauss_tmin_tmax_integral, conditional_integral)
    if result.shape.rank != 0:
        result = tf.gather(result, 0, axis=-1)
    return result


def generalized_gaussexptail_integral(limits, params, model):
    del model
    mu = params["mu"]
    sigmal = params["sigmal"]
    alphal = params["alphal"]
    sigmar = params["sigmar"]
    alphar = params["alphar"]

    lower, upper = limits._rect_limits_tf
    lower = lower[:, 0]
    upper = upper[:, 0]

    return generalized_gaussexptail_integral_func(
        mu=mu,
        sigmal=sigmal,
        alphal=alphal,
        sigmar=sigmar,
        alphar=alphar,
        lower=lower,
        upper=upper,
    )


@z.function(wraps="tensor", keepalive=True)
def generalized_gaussexptail_integral_func(mu, sigmal, alphal, sigmar, alphar, lower, upper):
    upper_of_lowerint = znp.minimum(mu, upper)
    integral_left = gaussexptail_integral_func(mu=mu, sigma=sigmal, alpha=alphal, lower=lower, upper=upper_of_lowerint)
    left = tf.where(tf.less(mu, lower), znp.zeros_like(integral_left), integral_left)

    lower_of_upperint = znp.maximum(mu, lower)
    integral_right = gaussexptail_integral_func(
        mu=mu, sigma=sigmar, alpha=-alphar, lower=lower_of_upperint, upper=upper
    )
    right = tf.where(tf.greater(mu, upper), znp.zeros_like(integral_right), integral_right)

    return left + right


class GaussExpTail(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alpha: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "GaussExpTail",
        label: str | None = None,
    ):
        """GaussExpTail shaped PDF. A combination of a Gaussian with an exponential tail on one side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha) =  \\begin{cases} \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & \\mbox{for }\\frac{x - \\mu}{\\sigma} \\geqslant -\\alpha \\newline
            \\exp{\\left(\\frac{|\\alpha|^2}{2} + |\\alpha|  \\left(\\frac{x - \\mu}{\\sigma}\\right)\\right)},
            & \\mbox{for }\\frac{x - \\mu}{\\sigma} < -\\alpha \\end{cases}

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alpha: parameter where to switch from a gaussian to the expontial tail
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
        params = {"mu": mu, "sigma": sigma, "alpha": alpha}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigma = params["sigma"]
        alpha = params["alpha"]
        x = x[0]
        return gaussexptail_func(x=x, mu=mu, sigma=sigma, alpha=alpha)


class GaussExpTailPDFRepr(BasePDFRepr):
    _implementation = GaussExpTail
    hs3_type: Literal["GaussExpTail"] = pydantic.Field("GaussExpTail", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alpha: Serializer.types.ParamTypeDiscriminated


gaussexptail_integral_limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))

GaussExpTail.register_analytic_integral(func=gaussexptail_integral, limits=gaussexptail_integral_limits)


class GeneralizedGaussExpTail(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "GeneralizedGaussExpTail",
        label: str | None = None,
    ):
        """GeneralizedGaussedExpTail shaped PDF which is Generalized assymetric double-sided GaussExpTail shaped PDF. A
        combination of two GaussExpTail using the **mu** (not a frac) and a different **sigma** on each side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma_{L}, \\alpha_{L}, \\sigma_{R}, \\alpha_{R}) =  \\begin{cases}
            \\exp{\\left(\\frac{|\\alpha_{L}|^2}{2} + |\\alpha_{L}|  \\left(\\frac{x - \\mu}{\\sigma_{L}}\\right)\\right)},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma_{L}} < -\\alpha_{L} \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma_{L}^2}),
            & \\mbox{for }-\\alpha_{L} \\leqslant \\frac{x - \\mu}{\\sigma_{L}} \\leqslant 0 \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma_{R}^2}),
            & \\mbox{for }0 \\leqslant \\frac{x - \\mu}{\\sigma_{R}} \\leqslant \\alpha_{R} \\newline
            \\exp{\\left(\\frac{|\\alpha_{R}|^2}{2} - |\\alpha_{R}|  \\left(\\frac{x - \\mu}{\\sigma_{R}}\\right)\\right)},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma_{R}} > \\alpha_{R}
            \\end{cases}

        Args:
            mu: The mean of the gaussian
            sigmal: Standard deviation of the gaussian on the left side
            alphal: parameter where to switch from a gaussian to the expontial tail on the left side
            sigmar: Standard deviation of the gaussian on the right side
            alphar: parameter where to switch from a gaussian to the expontial tail on the right side
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
        params = {
            "mu": mu,
            "sigmal": sigmal,
            "alphal": alphal,
            "sigmar": sigmar,
            "alphar": alphar,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigmal = params["sigmal"]
        alphal = params["alphal"]
        sigmar = params["sigmar"]
        alphar = params["alphar"]
        x = x[0]
        return generalized_gaussexptail_func(
            x=x,
            mu=mu,
            sigmal=sigmal,
            alphal=alphal,
            sigmar=sigmar,
            alphar=alphar,
        )


class GeneralizedGaussExpTailPDFRepr(BasePDFRepr):
    _implementation = GeneralizedGaussExpTail
    hs3_type: Literal["GeneralizedGaussExpTail"] = pydantic.Field("GeneralizedGaussExpTail", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigmal: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    sigmar: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated


GeneralizedGaussExpTail.register_analytic_integral(
    func=generalized_gaussexptail_integral, limits=gaussexptail_integral_limits
)


# TODO: calculations written as naive convertions from C++: u * u instead of tf.square -> fix in the future?
@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def landau_pdf_func(x, mu, sigma):
    """Calculate the Landau PDF.
    Args:
         x: value(s) for which the PDF will be calculated.
         mu: Mean value
         sigma: width

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Based on code from "https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html" (start line 21),
        i.e. LANDAU pdf : algorithm from CERNLIB G110 denlan, the same algorithm is used in GSL
        See also the paper
            Kölbig, Kurt Siegfried, and Benno Schorr. "A program package for the Landau distribution."
            Comput. Phys. Commun. 31.CERN-DD-83-18 (1983): 97-111.
    """
    # x = z.unstack_x(x)

    # define constant parameters
    p1 = znp.array([0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253])
    q1 = znp.array([1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063])

    p2 = znp.array([0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211])
    q2 = znp.array([1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714])

    p3 = znp.array([0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101])
    q3 = znp.array([1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675])

    p4 = znp.array([0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186])
    q4 = znp.array([1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511])

    p5 = znp.array([1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910])
    q5 = znp.array([1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357])

    p6 = znp.array([1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109])
    q6 = znp.array([1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939])

    a1 = znp.array([0.04166666667, -0.01996527778, 0.02709538966])
    a2 = znp.array([-1.845568670, -4.284640743])

    tf.assert_greater(sigma, znp.array(0.0), message="Landau PDF: sigma must be > 0")

    v = (x - mu) / sigma
    # denlan = 0.0

    # the thresholds set for the different cases
    THRESHOLDS_PDF = znp.array([-5.5, -1.0, 1.0, 5.0, 12.0, 50.0, 300.0])

    # the different function definitions for each case : how to process v depending on its value
    def case_less_than_minus5_5_PDF(v):
        u = znp.exp(v + 1.0)
        ue = znp.exp(-1 / u)
        us = znp.sqrt(u)

        greaterval = 0.3989422803 * (ue / us) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u)
        return znp.where(
            u < 1e-10,
            znp.ones_like(greaterval),
            greaterval,
        )

    def case_less_than_minus1_PDF(v):
        u = znp.exp(-v - 1)
        return (
            znp.exp(-u)
            * znp.sqrt(u)
            * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v)
            / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v)
        )

    def case_less_than_1_PDF(v):
        return (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / (
            q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v
        )

    def case_less_than_5_PDF(v):
        return (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / (
            q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v
        )

    def case_less_than_12_PDF(v):
        u = 1 / v
        return (
            u
            * u
            * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u)
            / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
        )

    def case_less_than_50_PDF(v):
        u = 1 / v
        return (
            u
            * u
            * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u)
            / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
        )

    def case_less_than_300_PDF(v):
        u = 1 / v
        return (
            u
            * u
            * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u)
            / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
        )

    def default_case(v):
        u = 1 / (v - v * znp.log(v) / (v + 1))
        return u * u * (1 + (a2[0] + a2[1] * u) * u)

    # this is the equivalent of a long line of if / elif conditions.
    # exlusive != False
    # denlan = tf.case(
    #     [
    #         (tf.less(v, THRESHOLDS_PDF[0]), case_less_than_minus5_5_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[1]), case_less_than_minus1_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[2]), case_less_than_1_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[3]), case_less_than_5_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[4]), case_less_than_12_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[5]), case_less_than_50_PDF),
    #         (tf.less(v, THRESHOLDS_PDF[6]), case_less_than_300_PDF),
    #     ],
    #     default=default_case,
    #     exclusive=False,
    # )
    lt0 = v < THRESHOLDS_PDF[0]
    val1 = znp.where(lt0, case_less_than_minus5_5_PDF(v), default_case(v))
    lt1 = v < THRESHOLDS_PDF[1]
    cond2 = znp.logical_and(znp.logical_not(lt0), lt1)
    val2 = znp.where(cond2, case_less_than_minus1_PDF(v), val1)

    lt2 = v < THRESHOLDS_PDF[2]
    cond3 = znp.logical_and(znp.logical_not(lt1), lt2)
    val3 = znp.where(cond3, case_less_than_1_PDF(v), val2)
    lt3 = v < THRESHOLDS_PDF[3]
    cond4 = znp.logical_and(znp.logical_not(lt2), lt3)
    val4 = znp.where(cond4, case_less_than_5_PDF(v), val3)
    lt4 = v < THRESHOLDS_PDF[4]
    cond5 = znp.logical_and(znp.logical_not(lt3), lt4)
    val5 = znp.where(cond5, case_less_than_12_PDF(v), val4)
    lt5 = v < THRESHOLDS_PDF[5]
    cond6 = znp.logical_and(znp.logical_not(lt4), lt5)
    val6 = znp.where(cond6, case_less_than_50_PDF(v), val5)
    lt6 = v < THRESHOLDS_PDF[6]
    cond7 = znp.logical_and(znp.logical_not(lt5), lt6)
    val7 = znp.where(cond7, case_less_than_300_PDF(v), val6)

    return val7 / sigma


@z.function(wraps="tensor", keepalive=True)
def landau_cdf_func(x, mu, sigma):
    """Analytical function for the CDF of the Landau distribution.

    Args:
         x: value(s) for which the CDF will be calculated.
         mu: Mean value
         sigma: width

    Returns:
        `tf.Tensor`: The calculated CDF values.

    Notes:
        Based on code from this https://root.cern/doc/v610/ProbFuncMathCore_8cxx_source.html, start line 336.
    """
    # x = z.unstack_x(x)
    # define constant parameters
    p1 = znp.array([0.2514091491e0, -0.6250580444e-1, 0.1458381230e-1, -0.2108817737e-2, 0.7411247290e-3])
    q1 = znp.array([1.0, -0.5571175625e-2, 0.6225310236e-1, -0.3137378427e-2, 0.1931496439e-2])

    p2 = znp.array([0.2868328584e0, 0.3564363231e0, 0.1523518695e0, 0.2251304883e-1])
    q2 = znp.array([1.0, 0.6191136137e0, 0.1720721448e0, 0.2278594771e-1])

    p3 = znp.array([0.2868329066e0, 0.3003828436e0, 0.9950951941e-1, 0.8733827185e-2])
    q3 = znp.array([1.0, 0.4237190502e0, 0.1095631512e0, 0.8693851567e-2])

    p4 = znp.array([0.1000351630e1, 0.4503592498e1, 0.1085883880e2, 0.7536052269e1])
    q4 = znp.array([1.0, 0.5539969678e1, 0.1933581111e2, 0.2721321508e2])

    p5 = znp.array([0.1000006517e1, 0.4909414111e2, 0.8505544753e2, 0.1532153455e3])
    q5 = znp.array([1.0, 0.5009928881e2, 0.1399819104e3, 0.4200002909e3])

    p6 = znp.array([0.1000000983e1, 0.1329868456e3, 0.9162149244e3, -0.9605054274e3])
    q6 = znp.array([1.0, 0.1339887843e3, 0.1055990413e4, 0.5532224619e3])

    a1 = znp.array([0, -0.4583333333e0, 0.6675347222e0, -0.1641741416e1])
    a2 = znp.array([0, 1.0, -0.4227843351e0, -0.2043403138e1])

    v = (x - mu) / sigma

    THRESHOLDS_CDF = znp.array([-5.5, -1.0, 4.0, 12.0, 50.0, 300.0])

    def case_less_than_minus5_5_CDF(v):
        u = znp.exp(v + 1)
        return 0.3989422803 * znp.exp(-1.0 / u) * znp.sqrt(u) * (1 + (a1[1] + (a1[2] + a1[3] * u) * u) * u)

    def case_less_than_minus1_CDF(v):
        u = znp.exp(-v - 1)
        return (
            (znp.exp(-u) / znp.sqrt(u))
            * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v)
            / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v)
        )

    def case_less_than_1_CDF(v):
        return (p2[0] + (p2[1] + (p2[2] + p2[3] * v) * v) * v) / (q2[0] + (q2[1] + (q2[2] + q2[3] * v) * v) * v)

    def case_less_than_4_CDF(v):
        return (p3[0] + (p3[1] + (p3[2] + p3[3] * v) * v) * v) / (q3[0] + (q3[1] + (q3[2] + q3[3] * v) * v) * v)

    def case_less_than_12_CDF(v):
        u = 1.0 / v
        return (p4[0] + (p4[1] + (p4[2] + p4[3] * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + q4[3] * u) * u) * u)

    def case_less_than_50_CDF(v):
        u = 1.0 / v
        return (p5[0] + (p5[1] + (p5[2] + p5[3] * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + q5[3] * u) * u) * u)

    def case_less_than_300_CDF(v):
        u = 1.0 / v
        return (p6[0] + (p6[1] + (p6[2] + p6[3] * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + q6[3] * u) * u) * u)

    def default_case_CDF(v):
        u = 1.0 / (v - v * znp.log(v) / (v + 1))
        return 1 - (a2[1] + (a2[2] + a2[3] * u) * u) * u

    return tf.case(
        [
            (tf.less(v, THRESHOLDS_CDF[0]), case_less_than_minus5_5_CDF),
            (tf.less(v, THRESHOLDS_CDF[1]), case_less_than_minus1_CDF),
            (tf.less(v, THRESHOLDS_CDF[2]), case_less_than_1_CDF),
            (tf.less(v, THRESHOLDS_CDF[3]), case_less_than_4_CDF),
            (tf.less(v, THRESHOLDS_CDF[4]), case_less_than_12_CDF),
            (tf.less(v, THRESHOLDS_CDF[5]), case_less_than_50_CDF),
            (tf.less(v, THRESHOLDS_CDF[6]), case_less_than_300_CDF),
        ],
        default=default_case_CDF,
        exclusive=False,
    )


@z.function(wraps="tensor", keepalive=True)
def landau_integral(limits: ztyping.SpaceType, params: dict) -> tf.Tensor:
    """Calculates the analytic integral of the Landau PDF.

    Args:
        limits: An object with attribute rect_limits.
        params: A hashmap from which the parameters that defines the PDF will be extracted.
    Returns:
        The calculated integral.
    """
    lower, upper = limits.rect_limits
    lower_cdf = landau_cdf_func(x=lower, mu=params["mu"], sigma=params["sigma"])
    upper_cdf = landau_cdf_func(x=upper, mu=params["m"], sigma=params["sigma"])
    return upper_cdf - lower_cdf


class Landau(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Landau",
        label: str | None = None,
    ):
        """Landau distribution. Useful for describing energy loss of charged particles in thin layers.

        Formula for PDF and CDF are based on https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html and
        https://root.cern/doc/v610/ProbFuncMathCore_8cxx_source.html, respectively.

        These in turn come from CERNLIB G110 denlan, the same algorithm is used in GSL. The implementations seem
        to be implementations of:
            Kölbig, Kurt Siegfried, and Benno Schorr. "A program package for the Landau distribution."
            Comput. Phys. Commun. 31.CERN-DD-83-18 (1983): 97-111.

        Args:
            mu: the average value
            sigma: the width of the distribution
        """
        params = {"mu": mu, "sigma": sigma}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        del norm
        mu = params["mu"]
        sigma = params["sigma"]
        x = z.unstack_x(x)
        return landau_pdf_func(x, mu=mu, sigma=sigma)


class LandauPDFRepr(BasePDFRepr):
    _implementation = Landau
    hs3_type: Literal["Landau"] = pydantic.Field("Landau", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated


# These lines of code add the analytic integral function to Landau PDF.
# landau_integral_limits = Space(axes=(0,), limits=(ANY_LOWER, ANY_UPPER))
# Landau.register_analytic_integral(func=landau_integral, limits=landau_integral_limits)

#  Copyright (c) 2023 zfit

from __future__ import annotations

import numpy as np
import pydantic
import tensorflow as tf

from typing import Literal

import zfit.z.numpy as znp
from zfit import z
from ..core.basepdf import BasePDF
from ..core.serialmixin import SerializableMixin
from ..core.space import ANY_LOWER, ANY_UPPER, Space
from ..serialization import SpaceRepr, Serializer
from ..serialization.pdfrepr import BasePDFRepr
from ..util import ztyping
from ..util.ztyping import ExtendedInputType, NormInputType


def _powerlaw(x, a, k):
    return a * znp.power(x, k)


@z.function(wraps="zfit_tensor")
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
    func = znp.maximum(func, znp.zeros_like(func))
    return func


@z.function(wraps="zfit_tensor", stateless_args=False)
def double_crystalball_func(x, mu, sigma, alphal, nl, alphar, nr):
    cond = tf.less(x, mu)

    func = tf.where(
        cond,
        crystalball_func(x, mu, sigma, alphal, nl),
        crystalball_func(x, mu, sigma, -alphar, nr),
    )

    return func


# created with the help of TensorFlow autograph used on python code converted from ShapeCB of RooFit
def crystalball_integral(limits, params, model):
    mu = params["mu"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    n = params["n"]

    lower, upper = limits._rect_limits_tf

    integral = crystalball_integral_func(mu, sigma, alpha, n, lower, upper)
    return integral


@z.function(wraps="zfit_tensor")
# @tf.function  # BUG? TODO: problem with tf.function and input signature
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
    tmax, tmin = znp.where(alpha_negative, -tmin, tmax), znp.where(
        alpha_negative, -tmax, tmin
    )

    if_true_4 = (
        abs_sigma
        * sqrt_pi_over_two
        * (tf.math.erf(tmax / sqrt2) - tf.math.erf(tmin / sqrt2))
    )

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
        * (
            1.0 / znp.power(safe_b_tmin_ones, n - 1.0)
            - 1.0 / znp.power(safe_b_tmax_ones, n - 1.0)
        )
    )

    if_true_3 = tf.where(use_log, if_true_1, if_false_1)

    if_true_2 = a * abs_sigma * (znp.log(safe_b_tmin_ones) - znp.log(n / abs_alpha))
    if_false_2 = (
        a
        * abs_sigma
        / (1.0 - n)
        * (
            1.0 / znp.power(safe_b_tmin_ones, n - 1.0)
            - 1.0 / znp.power(n / abs_alpha, n - 1.0)
        )
    )
    term1 = tf.where(use_log, if_true_2, if_false_2)
    term2 = (
        abs_sigma
        * sqrt_pi_over_two
        * (tf.math.erf(tmax / sqrt2) - tf.math.erf(-abs_alpha / sqrt2))
    )
    if_false_3 = term1 + term2

    if_false_4 = tf.where(tf.less_equal(tmax, -abs_alpha), if_true_3, if_false_3)

    # if_false_4()
    result = tf.where(tf.greater_equal(tmin, -abs_alpha), if_true_4, if_false_4)
    if not result.shape.rank == 0:
        result = tf.gather(result, 0, axis=-1)  # remove last dim, should vanish
    return result


def double_crystalball_mu_integral(limits, params, model):
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


@z.function(wraps="zfit_tensor")
def double_crystalball_mu_integral_func(
    mu, sigma, alphal, nl, alphar, nr, lower, upper
):
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
    right = tf.where(
        tf.greater(mu, upper), znp.zeros_like(integral_right), integral_right
    )

    integral = left + right
    return integral


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
    ):
        """Crystal Ball shaped PDF. A combination of a Gaussian with a powerlaw tail.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha, n) =  \\begin{cases} \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & \\mbox{for}\\frac{x - \\mu}{\\sigma} \\geqslant -\\alpha \\newline
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

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|

        .. _CBShape: https://en.wikipedia.org/wiki/Crystal_Ball_function
        """
        params = {"mu": mu, "sigma": sigma, "alpha": alpha, "n": n}
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )

    def _unnormalized_pdf(self, x):
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        alpha = self.params["alpha"]
        n = self.params["n"]
        x = x.unstack_x()
        return crystalball_func(x=x, mu=mu, sigma=sigma, alpha=alpha, n=n)


class CrystalBallPDFRepr(BasePDFRepr):
    _implementation = CrystalBall
    hs3_type: Literal["CrystalBall"] = pydantic.Field("CrystalBall", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alpha: Serializer.types.ParamTypeDiscriminated
    n: Serializer.types.ParamTypeDiscriminated


crystalball_integral_limits = Space(
    axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),))
)

CrystalBall.register_analytic_integral(
    func=crystalball_integral, limits=crystalball_integral_limits
)


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
    ):
        """Double-sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a frac) on each side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha_{L}, n_{L}, \\alpha_{R}, n_{R}) =  \\begin{cases}
            A_{L} \\cdot (B_{L} - \\frac{x - \\mu}{\\sigma})^{-n},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma} < -\\alpha_{L} \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & -\\alpha_{L} \\leqslant \\mbox{for}\\frac{x - \\mu}{\\sigma} \\leqslant \\alpha_{R} \\newline
            A_{R} \\cdot (B_{R} - \\frac{x - \\mu}{\\sigma})^{-n},
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

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
        """
        params = {
            "mu": mu,
            "sigma": sigma,
            "alphal": alphal,
            "nl": nl,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )

    def _unnormalized_pdf(self, x):
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        alphal = self.params["alphal"]
        nl = self.params["nl"]
        alphar = self.params["alphar"]
        nr = self.params["nr"]
        x = x.unstack_x()
        return double_crystalball_func(
            x=x, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr
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


DoubleCB.register_analytic_integral(
    func=double_crystalball_mu_integral, limits=crystalball_integral_limits
)

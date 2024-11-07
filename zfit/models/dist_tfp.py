"""A rich selection of analytically implemented Distributions (models) are available in `TensorFlow Probability.

<https://github.com/tensorflow/probability>`_. While their API is slightly different from the zfit models, it is similar
enough to be easily wrapped.

Therefore, a convenient wrapper as well as a lot of implementations are provided.
"""

#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import Literal

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from pydantic.v1 import Field

import zfit.z.numpy as znp
from zfit import z
from zfit.util.exception import AnalyticSamplingNotImplemented

from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData
from ..core.parameter import convert_to_parameter
from ..core.serialmixin import SerializableMixin
from ..core.space import Space, supports
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..settings import ztypes
from ..util import ztyping
from ..util.deprecation import deprecated_args
from ..util.ztyping import ExtendedInputType, NormInputType

# TODO: improve? while loop over `.sample`? Maybe as a fallback if not implemented?


def tfd_analytic_sample(n: int, dist: tfd.Distribution, limits: ztyping.ObsTypeInput):
    """Sample analytically with a `tfd.Distribution` within the limits. No preprocessing.

    Args:
        n: Number of samples to get
        dist: Distribution to sample from
        limits: Limits to sample from within

    Returns:
        The sampled data with the number of samples and the number of observables.
    """
    lower_bound, upper_bound = limits.v0.limits  # not working with MultiSpace
    lower_prob_lim = dist.cdf(lower_bound)
    upper_prob_lim = dist.cdf(upper_bound)

    shape = (n, 1)
    prob_sample = z.random.uniform(shape=shape, minval=lower_prob_lim, maxval=upper_prob_lim)
    prob_sample.set_shape((None, 1))
    try:
        sample = dist.quantile(prob_sample)
    except NotImplementedError:
        raise AnalyticSamplingNotImplemented from None
    sample.set_shape((None, limits.n_obs))
    return sample


class WrapDistribution(BasePDF):  # TODO: extend functionality of wrapper, like icdf
    """Baseclass to wrap tensorflow-probability distributions automatically."""

    def __init__(
        self,
        distribution,
        dist_params,
        obs,
        params=None,
        dist_kwargs=None,
        dtype=ztypes.float,
        name=None,
        **kwargs,
    ):
        # Check if subclass of distribution?
        if dist_kwargs is None:
            dist_kwargs = {}

        if dist_params is None:
            dist_params = {}
        name = name or distribution.name
        params = dist_params.copy() if params is None else {k: convert_to_parameter(p) for k, p in params.items()}

        super().__init__(obs=obs, dtype=dtype, name=name, params=params, **kwargs)

        self._distribution = distribution
        self.dist_params = dist_params
        self.dist_kwargs = dist_kwargs
        self._inverse_analytic_integral = []

    @property
    def distribution(self):
        params = self.dist_params
        if callable(params):
            params = params()
        kwargs = self.dist_kwargs
        if callable(kwargs):
            kwargs = kwargs()
        return self._distribution(**params, **kwargs, name=self.name + "_tfp")

    def _unnormalized_pdf(self, x: ZfitData):
        value = z.unstack_x(x)  # TODO: use this? change shaping below?
        return self.distribution.prob(value=value, name="unnormalized_pdf")

    # TODO: register integral?
    @supports()
    def _analytic_integrate(self, limits, norm):
        del norm  # not supported
        lower, upper = limits._rect_limits_tf
        lower = z.unstack_x(lower)
        upper = z.unstack_x(upper)
        tf.debugging.assert_all_finite((lower, upper), "Are infinite limits needed? Causes troubles with NaNs")
        return self.distribution.cdf(upper) - self.distribution.cdf(lower)

    def _analytic_sample(self, n, limits: Space):
        return tfd_analytic_sample(n=n, dist=self.distribution, limits=limits)


# class KernelDensityTFP(WrapDistribution):
#
#     def __init__(self, loc: ztyping.ParamTypeInput, scale: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput,
#                  kernel: tfp.distributions.Distribution = tfp.distributions.Normal,
#                  weights: Union[None, np.ndarray, tf.Tensor] = None, name: str = "KernelDensity"):
#         """Kernel Density Estimation of loc and either a broadcasted or a per-loc scale with a Distribution as kernel.
#
#         Args:
#             loc: 1-D Tensor-like. The positions of the `kernel`. Determines how many kernels will be created.
#             scale: Broadcastable to the batch and event shape of the distribution. A scalar will simply broadcast
#                 to `loc` for a 1-D distribution.
#             obs: Observables
#             kernel: Distribution that is used as kernel
#             weights: Weights of each `loc`, can be None or Tensor-like with shape compatible with loc
#             name: Name of the PDF
#         """
#         if not isinstance(kernel,
#                           tfp.distributions.Distribution) and False:  # HACK remove False, why does test not work?
#             raise TypeError("Currently, only tfp distributions are supported as kernels. Please open an issue if this "
#                             "is too restrictive.")
#
#         if isinstance(loc, ZfitData):
#             if loc.weights is not None:
#                 if weights is not None:
#                     raise OverdefinedError("Cannot specify weights and use a `ZfitData` with weights.")
#                 else:
#                     weights = loc.weights
#
#         if weights is None:
#             weights = tf.ones_like(loc, dtype=tf.float64)
#         self._weights_loc = weights
#         self._weights_sum = z.reduce_sum(weights)
#         self._latent_loc = loc
#         params = {"scale": scale}
#         dist_params = {"loc": loc, "scale": scale}
#         super().__init__(distribution=kernel, dist_params=dist_params, obs=obs, params=params, dtype=ztypes.float,
#                          name=name)
#
#     def _unnormalized_pdf(self, x: "zfit.Data", norm_range=False):
#         value = znp.expand_dims(x.value(), -2)
#         new_shape = znp.concatenate([tf.shape(value)[:2], [tf.shape(self._latent_loc)[0], 4]], axis=0)
#         value = tf.broadcast_to(value, new_shape)
#         probs = self.distribution.prob(value=value, name="unnormalized_pdf")
#         # weights = znp.expand_dims(self._weights_loc, axis=-1)
#         weights = self._weights_loc
#         probs = z.reduce_sum(probs * weights, axis=-1) / self._weights_sum
#         return probs
#
#     @supports()
#     def _analytic_integrate(self, limits, norm_range):
#         lower, upper = limits.limits
#         if np.all(-np.array(lower) == np.array(upper)) and np.all(np.array(upper) == np.infty):
#             return z.reduce_sum(self._weights_loc)  # tfp distributions are normalized to 1
#         lower = z.to_real(lower[0], dtype=self.dtype)
#         # lower = tf.broadcast_to(lower, shape=(tf.shape(self._latent_loc)[0], limits.n_obs,))  # remove
#         upper = z.to_real(upper[0], dtype=self.dtype)
#         integral = self.distribution.cdf(upper) - self.distribution.cdf(lower)
#         integral = z.reduce_sum(integral * self._weights_loc, axis=-1) / self._weights_sum
#         return integral  # TODO: generalize for VectorSpaces


class Gauss(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Gauss",
        label=None,
    ):
        """Gaussian or Normal distribution with a mean (mu) and a standartdeviation (sigma).

        The gaussian shape is defined as

        .. math::
            f(x \\mid \\mu, \\sigma^2) = e^{ -\\frac{(x - \\mu)^{2}}{2\\sigma^2} }

        with the normalization over [-inf, inf] of

        .. math::
            \\frac{1}{\\sqrt{2\\pi\\sigma^2} }

        The normalization changes for different normalization ranges

        Args:
            mu: Mean of the gaussian dist
            sigma: Standard deviation or spread of the gaussian
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
        mu, sigma = self._check_input_params_tfp(mu, sigma)
        params = {"mu": mu, "sigma": sigma}

        def dist_params():
            return {"loc": mu.value(), "scale": sigma.value()}

        distribution = tfp.distributions.Normal
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class GaussPDFRepr(BasePDFRepr):
    _implementation = Gauss
    hs3_type: Literal["Gauss"] = Field("Gauss", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamInputTypeDiscriminated
    sigma: Serializer.types.ParamInputTypeDiscriminated


class ExponentialTFP(WrapDistribution):
    _N_OBS = 1

    def __init__(
        self,
        tau: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        name: str = "Exponential",
    ):
        (tau,) = self._check_input_params_tfp(tau)
        params = {"tau", tau}
        dist_params = {"rate": tau}
        distribution = tfp.distributions.Exponential
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
        )


class Uniform(WrapDistribution):
    _N_OBS = 1

    def __init__(
        self,
        low: ztyping.ParamTypeInput,
        high: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Uniform",
        label: str | None = None,
    ):
        """Uniform distribution which is constant between `low`, `high` and zero outside.

        Args:
            low: Below this value, the pdf is zero.
            high: Above this value, the pdf is zero.
            obs: Observables and normalization range the pdf is defined in
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
        low, high = self._check_input_params_tfp(low, high)
        params = {"low": low, "high": high}

        def dist_params():
            return {"low": low.value(), "high": high.value()}

        distribution = tfp.distributions.Uniform
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class TruncatedGauss(WrapDistribution):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        low: ztyping.ParamTypeInput,
        high: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "TruncatedGauss",
        label: str | None = None,
    ):
        """Gaussian distribution that is 0 outside of `low`, `high`. Equivalent to the product of Gauss and Uniform.

        Args:
            mu: Mean of the gaussian dist
            sigma: Standard deviation or spread of the gaussian
            low: Below this value, the pdf is zero.
            high: Above this value, the pdf is zero.
            obs: Observables and normalization range the pdf is defined in
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
        mu, sigma, low, high = self._check_input_params_tfp(mu, sigma, low, high)
        params = {"mu": mu, "sigma": sigma, "low": low, "high": high}
        distribution = tfp.distributions.TruncatedNormal

        def dist_params():
            return {
                "loc": mu.value(),
                "scale": sigma.value(),
                "low": low.value(),
                "high": high.value(),
            }

        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class Cauchy(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        m: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Cauchy",
        label: str | None = None,
    ):
        r"""Non-relativistic Breit-Wigner (Cauchy) PDF representing the energy distribution of a decaying particle.

        The (unnormalized) shape of the non-relativistic Breit-Wigner is given by

        .. math::

            \frac{1}{\gamma \left[1 + \left(\frac{x - m}{\gamma}\right)^2\right]}

        with :math:`m` the mean and :math:`\gamma` the width of the distribution.

        Args:
            m: Invariant mass of the unstable particle.
            gamma: Width of the shape.
            obs: Observables and normalization range the pdf is defined in
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
        m, gamma = self._check_input_params_tfp(m, gamma)
        params = {"m": m, "gamma": gamma}
        distribution = tfp.distributions.Cauchy

        def dist_params():
            return {"loc": m.value(), "scale": gamma.value()}

        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class CauchyPDFRepr(BasePDFRepr):
    _implementation = Cauchy
    hs3_type: Literal["Cauchy"] = Field("Cauchy", alias="type")
    x: SpaceRepr
    m: Serializer.types.ParamTypeDiscriminated
    gamma: Serializer.types.ParamTypeDiscriminated


class Poisson(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    @deprecated_args(None, "Use lam instead", "lamb")
    def __init__(
        self,
        lam: ztyping.ParamTypeInput = None,
        obs: ztyping.ObsTypeInput = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Poisson",
        lamb=None,
        label: str | None = None,
    ):
        """Poisson distribution, parametrized with an event rate parameter (lamb).

        The probability mass function of the Poisson distribution is given by

        .. math::
            f(x, \\lambda) = \\frac{\\lambda^{x}e^{-\\lambda}}{x!}

        Args:
            lamb: the event rate
            obs: Observables and normalization range the pdf is defined in
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
        if lamb is not None:
            lam = lamb
        del lamb
        (lam,) = self._check_input_params_tfp(lam)
        params = {"lam": lam}

        def dist_params():
            return {"rate": lam.value()}

        distribution = tfp.distributions.Poisson
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class PoissonPDFRepr(BasePDFRepr):
    _implementation = Poisson
    hs3_type: Literal["Poisson"] = Field("Poisson", alias="type")
    x: SpaceRepr
    lam: Serializer.types.ParamTypeDiscriminated


class LogNormal(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "LogNormal",
        label: str | None = None,
    ):
        r"""Log-normal distribution, the exponential of a normal distribution.

        The probability density function of the log-normal distribution is only defined for positive values and
        is given by

        .. math::
            f(x \\mid \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(\ln(x) - \mu)^2}{2\sigma^2}}

        with :math:`\mu` the mean and :math:`\sigma` the standard deviation of the underlying normal distribution.

        Args:
            mu: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.
            obs: Observables and normalization range the pdf is defined in
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
        mu, sigma = self._check_input_params_tfp(mu, sigma)

        params = {"mu": mu, "sigma": sigma}

        def dist_params():
            return {"loc": mu.value(), "scale": sigma.value()}

        distribution = tfp.distributions.LogNormal
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class LogNormalPDFRepr(BasePDFRepr):
    _implementation = LogNormal
    hs3_type: Literal["LogNormal"] = Field("LogNormal", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated


class ChiSquared(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        ndof: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "ChiSquared",
        label: str | None = None,
    ):
        """ChiSquared distribution for ndof degrees of freedom.

        The chisquared shape for `d` degrees of freedom is defined as

        .. math::

            f(x \\mid d) = x^(d/2 - 1) \\exp(-x/2) / Z

        with the normalization over [0, inf] of

        .. math::

            Z = \\frac{1}{2^{d/2} \\Gamma(d/2)}

        The normalization changes for different normalization ranges

        Args:
            ndof: Number of degrees of freedom
            obs: Observables and normalization range the pdf is defined in
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
        (ndof,) = self._check_input_params_tfp(ndof)
        params = {"ndof": ndof}

        def dist_params():
            return {"df": ndof.value()}

        distribution = tfp.distributions.Chi2
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class ChiSquaredPDFRepr(BasePDFRepr):
    _implementation = ChiSquared
    hs3_type: Literal["ChiSquared"] = Field("ChiSquared", alias="type")
    x: SpaceRepr
    ndof: Serializer.types.ParamTypeDiscriminated


class StudentT(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        ndof: ztyping.ParamTypeInput,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "StudentT",
        label: str | None = None,
    ):
        """StudentT distribution for ndof degrees of freedom.

        The StudentT shape for `d` degrees of freedom is defined as

        .. math::

            f(x \\mid d, \\mu, \\sigma) = \\left(1 + \\frac{1}{d} \\left(\\frac{x - \\mu}{\\sigma}\\right)^2\\right)^{-\\frac{d+1}{2}} / Z

        with the normalization over [-inf, inf] of

        .. math::

            Z = \\frac{\\sqrt{d \\pi} \\Gamma(\\frac{d}{2})}{\\Gamma(\\frac{d+1}{2})}

        The normalization changes for different normalization ranges

        Args:
            ndof: Number of degrees of freedom
            mu: Mean of the distribution
            sigma: Scale of the distribution
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
        ndof, mu, sigma = self._check_input_params_tfp(ndof, mu, sigma)
        params = {"ndof": ndof, "mu": mu, "sigma": sigma}

        def dist_params():
            return {"df": ndof.value(), "loc": mu.value(), "scale": sigma.value()}

        distribution = tfp.distributions.StudentT
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class StudentTPDFRepr(BasePDFRepr):
    _implementation = StudentT
    hs3_type: Literal["StudentT"] = Field("StudentT", alias="type")
    x: SpaceRepr
    ndof: Serializer.types.ParamTypeDiscriminated
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated


class QGauss(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        q: ztyping.ParamTypeInput,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "QGauss",
        label: str | None = None,
    ):
        """Q-Gaussian distribution with parameter `q`.

        The q-Gaussian is a probability distribution arising from the maximization of the Tsallis entropy under appropriate constraints.
        It is defined for q < 3 and the Gaussian distribution is recovered as q -> 1.
        For q < 1, is it the PDF of a bounded random variable.
        We only support 1 < q < 3 in this implementation.
        If you want to use exactly q = 1, use the `zfit.pdf.Gauss` class.
        During fitting, if you want to start from a Gaussian shape, you can initialize the `q` parameter to be really close to 1.
        It is related to the Student's t-distribution according to the `corresponding Wikipedia entry <https://en.wikipedia.org/wiki/Q-Gaussian_distribution#Student's_t-distribution>`_
        and that is how it is implemented here.

        The q-Gaussian shape for 1 < q < 3 is defined as

        .. math::

            f(x \\mid q, \\mu, \\sigma) = \\frac{1}{C_{q} \\sigma} e_{q}\\left(-\\left(\\frac{x - \\mu}{\\sigma}\\right)^{2}\\right)

        with

        .. math::

            e_q(x) = \\left[1 + (1 - q) x\\right]_{+}^{\\frac{1}{1 - q}}

        and the normalization over [-inf, inf] of

        .. math::

            C_{q} = \\frac{\\sqrt{\\pi} \\Gamma \\left(\\frac{3 - q}{2 (q - 1)}\\right)}{\\sqrt{q - 1}\\Gamma \\left(\\frac{1}{q - 1}\\right)}

        The normalization changes for different normalization ranges

        Args:
            q: Shape parameter of the q-Gaussian. Must be 1 < q < 3.
            mu: Mean of the distribution
            sigma: Scale of the distribution
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
        from zfit import run

        q, mu, sigma = self._check_input_params_tfp(q, mu, sigma)
        if run.executing_eagerly():
            if q < 1 or q > 3:
                msg = "q < 1 or q > 3 are not supported"
                raise ValueError(msg)
            if q == 1:
                msg = "q = 1 is a Gaussian, use Gauss instead."
                raise ValueError(msg)
        elif run.numeric_checks:
            tf.debugging.assert_greater(q, znp.asarray(1.0), "q must be > 1")
            tf.debugging.assert_less(q, znp.asarray(3.0), "q must be < 3")
        params = {"q": q, "mu": mu, "sigma": sigma}

        # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
        # relation to Student's t-distribution

        # 1/(2 sigma^2) = 1 / (3 - q)
        # 2 sigma^2 = 3 - q
        # sigma = sqrt((3 - q)/2)

        def dist_params(q=q, mu=mu, sigma=sigma):
            if run.numeric_checks:
                tf.debugging.assert_greater(q, znp.asarray(1.0), "q must be > 1")
                tf.debugging.assert_less(q, znp.asarray(3.0), "q must be < 3")
            df = (3 - q.value()) / (q.value() - 1)
            scale = sigma.value() / tf.sqrt(0.5 * (3 - q.value()))
            return {"df": df, "loc": mu.value(), "scale": scale}

        distribution = tfp.distributions.StudentT
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class QGaussPDFRepr(BasePDFRepr):
    _implementation = QGauss
    hs3_type: Literal["QGauss"] = Field("QGauss", alias="type")
    x: SpaceRepr
    q: Serializer.types.ParamTypeDiscriminated
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated


class BifurGauss(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "BifurGauss",
        label: str | None = None,
    ):
        """Bifurcated Gaussian distribution different standard deviations for the left and right side of the mean.

        The bifurcated Gaussian shape is defined as

        .. math::

            f(x \\mid \\mu, \\sigma_{L}, \\sigma_{R}) = \\begin{cases}
            A \\exp{\\left(-\\frac{(x - \\mu)^2}{2 \\sigma_{L}^2}\\right)}, & \\mbox{for } x < \\mu \\newline
            A \\exp{\\left(-\\frac{(x - \\mu)^2}{2 \\sigma_{R}^2}\\right)}, & \\mbox{for } x \\geq \\mu
            \\end{cases}

        with the normalization over [-inf, inf] of

        .. math::

            A = \\sqrt{\\frac{2}{\\pi}} \\frac{1}{\\sigma_{L} + \\sigma_{R}}

        The normalization changes for different normalization ranges

        Args:
            mu: Mean of the distribution
            sigmal: Standard deviation on the left side of the mean
            sigmar: Standard deviation for the right side of the mean
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
        mu, sigmal, sigmar = self._check_input_params_tfp(mu, sigmal, sigmar)
        params = {"mu": mu, "sigmal": sigmal, "sigmar": sigmar}

        # sigmal = scale / skewness
        # sigmar = scale * skewness
        # scale = sigmal * skewness
        # sigmar = sigmal * skewness^2
        # skewness = sqrt(sigmar / sigmal)
        # scale = sigmal * sqrt(sigmar / sigmal)

        def dist_params():
            scale = sigmal.value() * znp.sqrt(sigmar.value() / sigmal.value())
            skewness = znp.sqrt(sigmar.value() / sigmal.value())
            return {"loc": mu.value(), "scale": scale, "skewness": skewness}

        distribution = tfp.distributions.TwoPieceNormal
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class BifurGaussPDFRepr(BasePDFRepr):
    _implementation = BifurGauss
    hs3_type: Literal["BifurGauss"] = Field("BifurGauss", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigmal: Serializer.types.ParamTypeDiscriminated
    sigmar: Serializer.types.ParamTypeDiscriminated


class Gamma(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        gamma: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        mu: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Gamma",
        label: str | None = None,
    ):
        """Gamma distribution.

        The gamma shape is parametrized here with `gamma`, `beta` and `mu`, following
        the same parametrization `as RooFit <https://root.cern.ch/doc/master/classRooGamma.html>`_.
        The gamma shape is defined as

        .. math::

            f(x \\mid \\gamma, \\beta, \\mu) = (x - \\mu)^{\\gamma - 1} \\exp{\\left(-\\frac{x - \\mu}{\\beta}\\right)} / Z

        with the normalization over [0, inf] of

        .. math::

            Z = \\Gamma(\\gamma) \\beta^{\\gamma}

        The normalization changes for different normalization ranges and `Z=1` for the unnormalized shape.

        Args:
            gamma: Shape parameter of the gamma distribution
            beta: Scale parameter of the gamma distribution
            mu: Shift of the distribution
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
        gamma, beta, mu = self._check_input_params_tfp(gamma, beta, mu)
        params = {"gamma": gamma, "beta": beta, "mu": mu}

        def dist_params():
            return {"concentration": gamma.value(), "rate": 1 / beta.value(), "loc": mu.value()}

        def distribution(concentration, rate, loc, name):
            return tfd.TransformedDistribution(
                distribution=tfp.distributions.Gamma(concentration, rate),
                bijector=tfp.bijectors.Shift(loc),
                name=name,
            )

        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class GammaPDFRepr(BasePDFRepr):
    _implementation = Gamma
    hs3_type: Literal["Gamma"] = Field("Gamma", alias="type")
    x: SpaceRepr
    gamma: Serializer.types.ParamTypeDiscriminated
    beta: Serializer.types.ParamTypeDiscriminated
    mu: Serializer.types.ParamTypeDiscriminated


class JohnsonSU(WrapDistribution, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        lambd: ztyping.ParamTypeInput,
        gamma: ztyping.ParamTypeInput,
        delta: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "JohnsonSU",
        label: str | None = None,
    ):
        """Johnson's SU distribution.

        The Johnson SU shape is parametrized here with `mu`, `lambd`, `gamma` and `delta`, following
        the same parametrization `as RooFit <https://root.cern.ch/doc/master/classRooJohnson.html>`_.
        The Johnson SU shape results from transforming a normally distributed variable `x` to

        .. math::

            z = \\gamma + \\delta \\sinh^{-1}\\left(\\frac{x - \\mu}{\\lambda}\\right)

        The resulting shape is then

        .. math::

            f(x \\mid \\mu, \\lambda, \\gamma, \\delta) = \\exp{\\left[-\\frac{1}{2} \\left(\\gamma + \\delta \\sinh^{-1}\\left(\\frac{x - \\mu}{\\lambda}\\right)\\right)^2\\right]} / Z

        with the normalization over [-inf, inf] of

        .. math::

            Z = \\lambda \\sqrt{2 \\pi} \\sqrt{1 + \\left(\\frac{x - \\mu}{\\lambda}\\right)^2} / \\delta

        The normalization changes for different normalization ranges and `Z=1` for the unnormalized shape.

        Args:
            mu: Mean of the distribution
            lambd: Scale of the distribution
            gamma: Skewness of the distribution
            delta: Tailweight of the distribution
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

        mu, lambd, gamma, delta = self._check_input_params_tfp(mu, lambd, gamma, delta)
        params = {"mu": mu, "lambd": lambd, "gamma": gamma, "delta": delta}

        def dist_params():
            return {"skewness": gamma.value(), "tailweight": delta.value(), "loc": mu.value(), "scale": lambd.value()}

        distribution = tfp.distributions.JohnsonSU
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
            label=label,
        )


class JohnsonSUPDFRepr(BasePDFRepr):
    _implementation = JohnsonSU
    hs3_type: Literal["JohnsonSU"] = Field("JohnsonSU", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    lambd: Serializer.types.ParamTypeDiscriminated
    gamma: Serializer.types.ParamTypeDiscriminated
    delta: Serializer.types.ParamTypeDiscriminated

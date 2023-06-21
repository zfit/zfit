"""A rich selection of analytically implemented Distributions (models) are available in `TensorFlow Probability.

<https://github.com/tensorflow/probability>`_. While their API is slightly different from the zfit models, it is similar
enough to be easily wrapped.

Therefore, a convenient wrapper as well as a lot of implementations are provided.
"""
#  Copyright (c) 2023 zfit
from __future__ import annotations

from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from pydantic import Field

from typing import Literal

from zfit import z
from zfit.util.exception import (
    AnalyticSamplingNotImplemented,
)
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData
from ..core.parameter import convert_to_parameter
from ..core.serialmixin import SerializableMixin
from ..core.space import Space, supports
from ..serialization import SpaceRepr, Serializer
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
    lower_bound, upper_bound = limits.rect_limits
    lower_prob_lim = dist.cdf(lower_bound)
    upper_prob_lim = dist.cdf(upper_bound)

    shape = (n, 1)
    prob_sample = z.random.uniform(
        shape=shape, minval=lower_prob_lim, maxval=upper_prob_lim
    )
    prob_sample.set_shape((None, 1))
    try:
        sample = dist.quantile(prob_sample)
    except NotImplementedError:
        raise AnalyticSamplingNotImplemented
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
        if params is None:
            params = OrderedDict((k, p) for k, p in dist_params.items())
        else:
            params = OrderedDict(
                (k, convert_to_parameter(p)) for k, p in params.items()
            )

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
        lower, upper = limits._rect_limits_tf
        lower = z.unstack_x(lower)
        upper = z.unstack_x(upper)
        tf.debugging.assert_all_finite(
            (lower, upper), "Are infinite limits needed? Causes troubles with NaNs"
        )
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
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Gauss",
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
            obs: Observables and normalization range the pdf is defined in
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
        """
        mu, sigma = self._check_input_params(mu, sigma)
        params = OrderedDict((("mu", mu), ("sigma", sigma)))
        dist_params = lambda: dict(loc=mu.value(), scale=sigma.value())
        distribution = tfp.distributions.Normal
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
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
        (tau,) = self._check_input_params(tau)
        params = OrderedDict((("tau", tau),))
        dist_params = dict(rate=tau)
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
            name: |@doc:model.init.pdf||@docend:model.init.pdf|
        """
        low, high = self._check_input_params(low, high)
        params = OrderedDict((("low", low), ("high", high)))
        dist_params = lambda: dict(low=low.value(), high=high.value())
        distribution = tfp.distributions.Uniform
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
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
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
        """
        mu, sigma, low, high = self._check_input_params(mu, sigma, low, high)
        params = OrderedDict(
            (("mu", mu), ("sigma", sigma), ("low", low), ("high", high))
        )
        distribution = tfp.distributions.TruncatedNormal
        dist_params = lambda: dict(
            loc=mu.value(), scale=sigma.value(), low=low.value(), high=high.value()
        )
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
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
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
        """
        m, gamma = self._check_input_params(m, gamma)
        params = OrderedDict((("m", m), ("gamma", gamma)))
        distribution = tfp.distributions.Cauchy
        dist_params = lambda: dict(loc=m.value(), scale=gamma.value())
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
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
            name: Name of the PDF
        """
        if lamb is not None:
            lam = lamb
        del lamb
        (lam,) = self._check_input_params(lam)
        params = {"lam": lam}
        dist_params = lambda: dict(rate=lam.value())
        distribution = tfp.distributions.Poisson
        super().__init__(
            distribution=distribution,
            dist_params=dist_params,
            obs=obs,
            params=params,
            name=name,
            extended=extended,
            norm=norm,
        )


class PoissonPDFRepr(BasePDFRepr):
    _implementation = Poisson
    hs3_type: Literal["Poisson"] = Field("Poisson", alias="type")
    x: SpaceRepr
    lam: Serializer.types.ParamTypeDiscriminated

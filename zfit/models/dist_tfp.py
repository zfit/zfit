"""
A rich selection of analytically implemented Distributions (models) are available in
`TensorFlow Probability <https://github.com/tensorflow/probability>`_. While their API is slightly
different from the zfit models, it is similar enough to be easily wrapped.

Therefore a convenient wrapper as well as a lot of implementations are provided.
"""
#  Copyright (c) 2019 zfit

from collections import OrderedDict
from typing import Union

import numpy as np

import tensorflow_probability as tfp

import tensorflow as tf

from zfit import ztf
from zfit.util.exception import OverdefinedError
from ..util import ztyping
from ..settings import ztypes
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitParameter, ZfitData
from ..core.limits import no_norm_range, supports
from ..core.parameter import convert_to_parameter


class WrapDistribution(BasePDF):  # TODO: extend functionality of wrapper, like icdf
    """Baseclass to wrap tensorflow-probability distributions automatically.

    """

    def __init__(self, distribution, dist_params, obs, params=None, dist_kwargs=None, dtype=ztypes.float, name=None,
                 **kwargs):
        # Check if subclass of distribution?
        if dist_kwargs is None:
            dist_kwargs = {}

        if dist_params is None:
            dist_params = {}
        name = name or distribution.name
        if params is None:
            params = OrderedDict((k, p) for k, p in dist_params.items())
        else:
            params = OrderedDict((k, convert_to_parameter(p)) for k, p in params.items())

        super().__init__(obs=obs, dtype=dtype, name=name, params=params, **kwargs)

        # self.tf_distribution = self.parameters['distribution']
        self._distribution = distribution
        self.dist_params = dist_params
        self.dist_kwargs = dist_kwargs

    @property
    def distribution(self):

        params = self.dist_params
        if callable(params):
            params = params()
        kwargs = self.dist_kwargs
        if callable(kwargs):
            kwargs = kwargs()
        return self._distribution(**params, **kwargs, name=self.name + "_tfp")

    def _unnormalized_pdf(self, x: "zfit.Data", norm_range=False):
        value = x.value()
        return tf.reshape(self.distribution.prob(value=value, name="unnormalized_pdf"),
                          shape=(-1,))  # TODO batch shape just removed

    # TODO: register integral
    @supports()
    def _analytic_integrate(self, limits, norm_range):
        lower, upper = limits.limits
        if np.all(-np.array(lower) == np.array(upper)) and np.all(np.array(upper) == np.infty):
            return ztf.to_real(1.)  # tfp distributions are normalized to 1
        lower = ztf.to_real(lower[0], dtype=self.dtype)
        upper = ztf.to_real(upper[0], dtype=self.dtype)
        integral = self.distribution.cdf(upper) - self.distribution.cdf(lower)
        return integral[0]


# class KernelDensity(WrapDistribution):
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
#         self._weights_sum = ztf.reduce_sum(weights)
#         self._latent_loc = loc
#         params = {"scale": scale}
#         dist_params = {"loc": loc, "scale": scale}
#         super().__init__(distribution=kernel, dist_params=dist_params, obs=obs, params=params, dtype=ztypes.float,
#                          name=name)
#
#     def _unnormalized_pdf(self, x: "zfit.Data", norm_range=False):
#         value = tf.expand_dims(x.value(), -2)
#         new_shape = tf.concat([tf.shape(value)[:2], [tf.shape(self._latent_loc)[0], 4]], axis=0)
#         value = tf.broadcast_to(value, new_shape)
#         probs = self.distribution.prob(value=value, name="unnormalized_pdf")
#         # weights = tf.expand_dims(self._weights_loc, axis=-1)
#         weights = self._weights_loc
#         probs = ztf.reduce_sum(probs * weights, axis=-1) / self._weights_sum
#         return probs
#
#     @supports()
#     def _analytic_integrate(self, limits, norm_range):
#         lower, upper = limits.limits
#         if np.all(-np.array(lower) == np.array(upper)) and np.all(np.array(upper) == np.infty):
#             return ztf.reduce_sum(self._weights_loc)  # tfp distributions are normalized to 1
#         lower = ztf.to_real(lower[0], dtype=self.dtype)
#         # lower = tf.broadcast_to(lower, shape=(tf.shape(self._latent_loc)[0], limits.n_obs,))  # remove
#         upper = ztf.to_real(upper[0], dtype=self.dtype)
#         integral = self.distribution.cdf(upper) - self.distribution.cdf(lower)
#         integral = ztf.reduce_sum(integral * self._weights_loc, axis=-1) / self._weights_sum
#         return integral  # TODO: generalize for VectorSpaces



class Gauss(WrapDistribution):
    _N_OBS = 1

    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput,
                 name: str = "Gauss"):
        """Gaussian or Normal distribution with a mean (mu) and a standartdevation (sigma).

        The gaussian shape is defined as

        .. math::
            f(x \mid \mu, \\sigma^2) = e^{ -\\frac{(x - \\mu)^{2}}{2\\sigma^2} }

        with the normalization over [-inf, inf] of

        .. math::
            \\frac{1}{\\sqrt{2\pi\sigma^2} }

        The normalization changes for different normalization ranges

        Args:
            mu (:py:class:`~zfit.Parameter`): Mean of the gaussian dist
            sigma (:py:class:`~zfit.Parameter`): Standard deviation or spread of the gaussian
            obs (:py:class:`~zfit.Space`): Observables and normalization range the pdf is defined in
            name (str): Name of the pdf
        """
        mu, sigma = self._check_input_params(mu, sigma)
        params = OrderedDict((('mu', mu), ('sigma', sigma)))
        dist_params = dict(loc=mu, scale=sigma)
        distribution = tfp.distributions.Normal
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, params=params, name=name + "_tfp")


class ExponentialTFP(WrapDistribution):
    _N_OBS = 1

    def __init__(self, tau: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput, name: str = "Exponential"):
        (tau,) = self._check_input_params(tau)
        params = OrderedDict((('tau', tau),))
        dist_params = dict(rate=tau)
        distribution = tfp.distributions.Exponential
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, params=params, name=name + "_tfp")


class Uniform(WrapDistribution):
    _N_OBS = 1

    def __init__(self, low: ztyping.ParamTypeInput, high: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput,
                 name: str = "Uniform"):
        """Uniform distribution which is constant between `low`, `high` and zero outside.

        Args:
            low (:py:class:`~zfit.Parameter`): Below this value, the pdf is zero.
            high (:py:class:`~zfit.Parameter`): Above this value, the pdf is zero.
            obs (:py:class:`~zfit.Space`): Observables and normalization range the pdf is defined in
            name (str): Name of the pdf
        """
        low, high = self._check_input_params(low, high)
        params = OrderedDict((("low", low), ("high", high)))
        dist_params = dict(low=low, high=high)
        distribution = tfp.distributions.Uniform
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, params=params, name=name + "_tfp")


class TruncatedGauss(WrapDistribution):
    _N_OBS = 1

    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput, low: ztyping.ParamTypeInput,
                 high: ztyping.ParamTypeInput, obs: ztyping.ObsTypeInput, name: str = "TruncatedGauss"):
        """Gaussian distribution that is 0 outside of `low`, `high`. Equivalent to the product of Gauss and Uniform.

        Args:
            mu (:py:class:`~zfit.Parameter`): Mean of the gaussian dist
            sigma (:py:class:`~zfit.Parameter`): Standard deviation or spread of the gaussian
            low (:py:class:`~zfit.Parameter`): Below this value, the pdf is zero.
            high (:py:class:`~zfit.Parameter`): Above this value, the pdf is zero.
            obs (:py:class:`~zfit.Space`): Observables and normalization range the pdf is defined in
            name (str): Name of the pdf
        """
        mu, sigma, low, high = self._check_input_params(mu, sigma, low, high)
        params = OrderedDict((("mu", mu), ("sigma", sigma), ("low", low), ("high", high)))
        distribution = tfp.distributions.TruncatedNormal
        dist_params = dict(loc=mu, scale=sigma, low=low, high=high)
        super().__init__(distribution=distribution, dist_params=dist_params,
                         obs=obs, params=params, name=name)


if __name__ == '__main__':
    exp1 = ExponentialTFP(tau=5., obs=['a'])

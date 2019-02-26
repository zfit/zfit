"""
A rich selection of analytically implemented Distributions (models) are available in
`TensorFlow Probability <https://github.com/tensorflow/probability>`_. While their API is slightly
different from the zfit models, it is similar enough to be easily wrapped.

Therefore a convenient wrapper as well as a lot of implementations are provided.
"""
from collections import OrderedDict

import numpy as np

import tensorflow_probability as tfp

import tensorflow as tf

from zfit import ztf
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitParameter
from ..core.limits import no_norm_range, supports
from ..core.parameter import convert_to_parameter


class WrapDistribution(BasePDF):  # TODO: extend functionality of wrapper, like icdf
    """Baseclass to wrap tensorflow-probability distributions automatically.

    """

    def __init__(self, distribution, obs, params=None, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        if params is None:  # auto extract from distribution
            params = OrderedDict((k, v) for k, v in distribution.parameters.items() if isinstance(v, ZfitParameter))
        else:
            params = OrderedDict((k, convert_to_parameter(p)) for k, p in params.items())

        super().__init__(obs=obs, dtype=distribution.dtype, name=name, params=params, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.distribution = distribution


    def _unnormalized_pdf(self, x: "zfit.data.Data", norm_range=False):
        value = x.unstack_x()
        return self.distribution.prob(value=value, name="unnormalized_pdf")  # TODO name

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


class Gauss(WrapDistribution):
    _N_OBS = 1

    def __init__(self, mu, sigma, obs, name="Gauss"):
        mu, sigma = self._check_input_params(mu, sigma)
        params = OrderedDict((('mu', mu), ('sigma', sigma)))
        distribution = tfp.distributions.Normal(loc=mu, scale=sigma, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, params=params, name=name)


class ExponentialTFP(WrapDistribution):
    _N_OBS = 1

    def __init__(self, tau, obs, name="Exponential"):
        (tau,) = self._check_input_params(tau)
        params = OrderedDict((('tau', tau),))
        distribution = tfp.distributions.Exponential(rate=tau, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, params=params, name=name)


class Uniform(WrapDistribution):
    _N_OBS = 1

    def __init__(self, low, high, obs, name="Uniform"):
        low, high = self._check_input_params(low, high)
        params = OrderedDict((("low", low), ("high", high)))
        distribution = tfp.distributions.Uniform(low=low, high=high, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, params=params, name=name)


if __name__ == '__main__':
    exp1 = ExponentialTFP(tau=5., obs=['a'])

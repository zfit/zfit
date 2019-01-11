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

    def __init__(self, distribution, obs, parameters=None, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        if parameters is None:  # auto extract from distribution
            parameters = OrderedDict((k, v) for k, v in distribution.parameters.items() if isinstance(v, ZfitParameter))
        else:
            parameters = OrderedDict((k, convert_to_parameter(p)) for k, p in parameters.items())

        super().__init__(obs=obs, dtype=distribution.dtype, name=name, parameters=parameters, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    @property
    def _n_dims(self):
        n_dims = self.tf_distribution.event_shape.as_list()
        n_dims = (n_dims or [1])[0]  # n_obs is a list
        return n_dims

    def _unnormalized_pdf(self, x: "zfit.core.data.Data", norm_range=False):
        return self.tf_distribution.prob(value=x.value(), name="unnormalized_pdf")  # TODO name

    # TODO: register integral
    @supports()
    def _analytic_integrate(self, limits, norm_range):
        lower, upper = limits.limits
        if all(-np.array(lower) == np.array(upper) == np.infty):
            return ztf.to_real(1.)  # tfp distributions are normalized to 1
        lower = ztf.to_real(lower[0], dtype=self.dtype)
        upper = ztf.to_real(upper[0], dtype=self.dtype)
        integral = self.tf_distribution.cdf(upper) - self.tf_distribution.cdf(lower)
        return integral


class Gauss(WrapDistribution):
    def __init__(self, mu, sigma, obs, name="Gauss"):
        mu, sigma = self._check_input_parameters(mu, sigma)
        parameters = OrderedDict((('mu', mu), ('sigma', sigma)))
        distribution = tfp.distributions.Normal(loc=mu, scale=sigma, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, parameters=parameters, name=name)


class Exponential(WrapDistribution):
    def __init__(self, tau, obs, name="Exponential"):
        (tau,) = self._check_input_parameters(tau)
        parameters = OrderedDict((('tau', tau),))
        distribution = tfp.distributions.Exponential(rate=tau, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, parameters=parameters, name=name)


class Uniform(WrapDistribution):
    def __init__(self, low, high, obs, name="Uniform"):
        low, high = self._check_input_parameters(low, high)
        parameters = OrderedDict((("low", low), ("high", high)))
        distribution = tfp.distributions.Uniform(low=low, high=high, name=name + "_tfp")
        super().__init__(distribution=distribution, obs=obs, parameters=parameters, name=name)


if __name__ == '__main__':
    exp1 = Exponential(tau=5., obs=['a'])

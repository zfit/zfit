
import tensorflow as tf
import tensorflow_probability as tfp

from zfit.core.basepdf import BasePDF
from zfit.core.limits import no_norm_range, supports


class WrapDistribution(BasePDF):
    """Baseclass to wrap tensorflow-probability distributions automatically.

    """

    def __init__(self, distribution, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        super(WrapDistribution, self).__init__(distribution=distribution, name=name, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    def _unnormalized_prob(self, x):
        return self.tf_distribution.prob(value=x, name="_unnormalized_prob")  # TODO name

    # TODO: register integral
    @supports()
    def _analytic_integrate(self, limits, norm_range):  # TODO deal with norm_range
        lower, upper = limits.get_boundaries()  # TODO: limits
        upper = tf.cast(upper, dtype=tf.float64)
        lower = tf.cast(lower, dtype=tf.float64)
        integral = self.tf_distribution.cdf(upper) - self.tf_distribution.cdf(lower)
        return integral


class Normal(WrapDistribution):
    def __init__(self, loc, scale, name="Normal"):
        distribution = tfp.distributions.Normal(loc=loc, scale=scale, name=name + "_tf")
        super(Normal, self).__init__(distribution=distribution, name=name)

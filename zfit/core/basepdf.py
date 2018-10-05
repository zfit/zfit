"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
import tensorflow_probability as tfp
import numpy as np

import zfit.core.utils as utils
import zfit.core.integrate
import zfit.settings


class AbstractBasePDF(object):

    def sample(self, sample_shape=(), seed=None, name='sample'):
        raise NotImplementedError

    def func(self, value, name='func'):
        raise NotImplementedError

    def log_prob(self, value, name='log_prob'):
        raise NotImplementedError

    def integrate(self, value, name='integrate'):
        self.error = NotImplementedError
        raise self.error

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        raise NotImplementedError

    def event_shape_tensor(self, name='event_shape_tensor'):
        raise NotImplementedError


class BasePDF(tf.distributions.Distribution, AbstractBasePDF):
    _DEFAULTS_integration = utils.dotdict()
    _DEFAULTS_integration.norm_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 10000
    _DEFAULTS_integration.numeric_integrate = zfit.core.integrate.numeric_integrate

    _analytic_integral = zfit.core.integrate.AnalyticIntegral()

    def __init__(self, name="BaseDistribution", **kwargs):
        # TODO: catch some args from kwargs that belong to the super init?
        super(BasePDF, self).__init__(dtype=zfit.settings.fptype, reparameterization_type=False,
                                      validate_args=True, parameters=kwargs,
                                      allow_nan_stats=False, name=name)

        self.norm_range = None
        # self.normalization_opt = {'n_draws': 10000000, 'range': (-100., 100.)}
        self._integration = utils.dotdict()
        self._integration.norm_sampler = self._DEFAULTS_integration.norm_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.numeric_integrate = self._DEFAULTS_numeric_integrate
        self._normalization_value = None

    def _func(self, value):
        raise NotImplementedError

    def _call_func(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = tf.convert_to_tensor(value, name="value")
            try:
                return self._func(value, **kwargs)
            except NotImplementedError:
                return self.prob(value) * self.NAME_NEEDED_YIELD

    def func(self, value, name="func"):  # TODO: rename to unnormalized_prob?
        return self._call_func(value, name)

    def _prob(self, value):
        pdf = self.func(value) / self.normalization(value)
        return pdf

    # def _normalization_sampler(self):
    #     lower, upper = self.normalization_opt['range']
    #     return tf.distributions.Uniform(lower, upper)

    def _call_normalization(self, value):
        # TODO: caching? alternative

        return self._normalization(value)

    @classmethod
    def register_analytic_integral(cls, func, dims=None):
        """

        Args:
            func ():
            dims (tuple(int)):

        Returns:

        """
        cls._analytic_integral.register(func=func, dims=dims)

    def _integrate(self, limits):
        # TODO: handle analytic and more general MC method integration
        # dim = tf.shape(value)

        # TODO: get limits properly
        # HACK
        lower, upper = limits
        # TODO: get dimensions properly
        dim = 1  # HACK
        n_samples = self._integration.draws_per_dim  # TODO: add times dim or so
        samples_normed = self._integration.norm_sampler(dim=dim, num_results=n_samples,
                                                        dtype=zfit.settings.fptype)
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        avg = tfp.monte_carlo.expectation(f=self.func, samples=samples)
        integral = avg * (upper - lower)
        return tf.cast(integral, dtype=zfit.settings.fptype)

        # integ.auto_integrate()

    def integrate(self, limits, name='integrate'):
        """Integrate over the **function**.

        Args:
            dim (iterable(int)): the dimensions to integrate.
        """

        integral = self._call_integrate(limits=limits)
        return integral

    def _call_integrate(self, limits):
        try:
            integral = self._analytic_integrate(limits)
        except NotImplementedError:
            max_dims = self._analytic_integral.max_dims
            if max_dims:
                def part_int(value):
                    return self._analytic_integral.integrate(value, limits=limits, dims=max_dims)

                integral = self._integration.numeric_integrate(func=part_int, limits=limits,
                                                               method="TODO", mc_sampler="TODO")
            else:
                integral = self._integrate(limits=limits)
        return integral

    def analytic_integrate(self, value):
        # TODO: get limits
        integral = self._analytic_integrate(value)

    def _analytic_integrate(self, value):
        # TODO: user implementation requested
        raise NotImplementedError

    def _partial_analytic_integrate(self, value, limits, dims):
        """Partial integral over dims.

        Args:
            value ():
            dims (tuple(int)): The dims to integrate over

        Returns:
            Tensor:

        Raises:
            NotImplementedError: if the function is not implemented

        """
        # TODO: implement meaningful, how communicate integrated, not integrated vars?
        self._analytic_integrals.integrate(value=value, limits=limits,
                                           dims=dims)  # Whatevernot arguments
        raise NotImplementedError

    def normalization(self, value):
        normalization = self._call_normalization(value)
        return normalization

    def _normalization(self, value):

        # TODO: multidim, more complicated range
        normalization_value = self.integrate(value)
        return normalization_value


def wrap_distribution(dist):
    """Wraps a tfp.distribution instance."""


class WrapDistribution(BasePDF):

    def __init__(self, distribution, name="WrappedTFDistribution", **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        super(WrapDistribution, self).__init__(distribution=distribution, name=name, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    def _func(self, value):
        return self.tf_distribution.prob(value=value, name="asdf")  # TODO name

    def _analytic_integrate(self, value):
        lower, upper = self.norm_range  # TODO: limits
        upper = tf.cast(upper, dtype=tf.float64)
        lower = tf.cast(lower, dtype=tf.float64)
        integral = self.tf_distribution.cdf(upper, name="asdf2") - self.tf_distribution.cdf(lower,
                                                                                            name="asdf3")  # TODO name
        return integral


# TODO: remove below, play around while developing
if __name__ == "__main":
    import zfit

    mu_true = 1.4
    sigma_true = 1.8


    class TestGaussian(zfit.core.basepdf.BasePDF):
        def _func(self, value):
            return tf.exp(-(value - mu_true) ** 2 / sigma_true ** 2)  # non-normalized gaussian


    dist1 = TestGaussian()
    tf_gauss1 = tf.distributions.Normal(loc=mu_true, scale=sigma_true)
    wrapped = WrapDistribution(tf_gauss1)

    with tf.Session() as sess:
        res = sess.run(dist1.event_shape_tensor())
        print(res)

"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
import tensorflow_probability as tfp
import numpy as np

import zfit.core.utils as utils
import zfit.core.integrate as integ


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

    def __init__(self, name="BaseDistribution", **kwargs):
        # TODO: catch some args from kwargs that belong to the super init?
        super(BasePDF, self).__init__(dtype=tf.float64, reparameterization_type=False,
                                      validate_args=True, parameters=kwargs,
                                      allow_nan_stats=False, name=name)

        self.norm_range = None
        # self.normalization_opt = {'n_draws': 10000000, 'range': (-100., 100.)}
        self._integration = utils.dotdict()
        self._integration.norm_sampler = self._DEFAULTS_integration.norm_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
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

    def func(self, value, name="func"):
        return self._call_func(value, name)

    def _prob(self, value):
        pdf = self.func(value) / self.normalization(value)
        return pdf

    def _normalization_sampler(self):
        lower, upper = self.normalization_opt['range']
        return tf.distributions.Uniform(lower, upper)

    def _call_normalization(self, value):
        # TODO: caching?

        return self._normalization(value)

    def _integrate(self, value):
        # TODO: handle analytic and more general MC method integration
        # dim = tf.shape(value)

        # TODO: get limits properly
        # HACK
        lower, upper = self.norm_range
        # TODO: get dimensions properly
        dim = 1  # HACK
        n_samples = self._integration.draws_per_dim  # TODO: add times dim or so
        samples_normed = self._integration.norm_sampler(dim=dim, num_results=n_samples)
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        avg = tfp.monte_carlo.expectation(f=self.func, samples=samples)
        integral = avg * (upper - lower)
        return integral

        # integ.auto_integrate()

    def integrate(self, value, name='integrate'):
        integral = self._call_integrate(value)
        return integral

    def _call_integrate(self, value):
        try:
            integral = self._integrate(value)
        except NotImplementedError:
            raise NotImplementedError("alternative not yet implemented.")
        return integral

    def analytic_integrate(self, value):
        # TODO: get limits
        integral = self.TODO

    def _analytic_integrate(self, value):
        # TODO: user implementation requested
        pass

    def _partial_analytic_integral(self, value):
        # TODO: user implementation requested
        return None

    def _partial_analytic_integrate(self, value):
        # TODO: implement meaningful, how communicate integrated, not integrated vars?
        part_integral, is_integrated = self._partial_analytic_integral(value)

    def normalization(self, value):
        normalization = self._call_normalization(value)
        return normalization

    def _normalization(self, value):

        # TODO: multidim, more complicated range
        normalization_value = self.integrate(value)
        return normalization_value


# TODO: remove below, play around while developing
if __name__ == "__main":
    import zfit
    class TestGaussian(zfit.core.basepdf.BasePDF):
        def _func(self, value):
            return tf.exp(-(value - 1.4) ** 2 / 1.8 ** 2)  # non-normalized gaussian

    dist1 = TestGaussian()

    with tf.Session() as sess:
        res = sess.run(dist1.event_shape_tensor())
        print(res)

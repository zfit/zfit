"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np


class AbstractBaseDistribution(object):

    def sample(self, sample_shape=(), seed=None, name='sample'):
        raise NotImplementedError

    def log_prob(self, value, name='log_prob'):
        raise NotImplementedError

    def integral(self, name='integral'):
        self.error = NotImplementedError
        raise self.error

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        raise NotImplementedError

    def event_shape_tensor(self, name='event_shape_tensor'):
        raise NotImplementedError


class BaseDistribution(tf.distributions.Distribution, AbstractBaseDistribution):

    def __init__(self, name="BaseDistribution"):
        super(BaseDistribution, self).__init__(dtype=None, reparameterization_type=False,
                                               validate_args=True,
                                               allow_nan_stats=False, name=name)

        self.normalization_opt = {'n_draws': 10000000, 'range': (-100., 100.)}
        self._normalization_value = None

    def _func(self, value):
        raise NotImplementedError

    def _call_func(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = tf.convert_to_tensor(value, name="value")
            try:
                return self._func(value, **kwargs)
            except NotImplementedError:
                return self.prob(value) * self.normalization()

    def func(self, value, name="func"):
        return self._call_func(value, name)

    def _prob(self, value):
        pdf = self.func(value) / self.normalization()
        return pdf

    def _normalization_sampler(self):
        lower, upper = self.normalization_opt['range']
        return tf.distributions.Uniform(lower, upper)

    def _set_normalization(self):
        x_sample = self._normalization_sampler().sample(self.normalization_opt['n_draws'])
        avg = tf.reduce_mean(self.func(x_sample))
        lower, upper = self.normalization_opt['range']
        self._normalization_value = avg * (upper - lower)
        return self._normalization_value

    def normalization(self):
        if self._normalization_value is None:
            self._set_normalization()
        return self._normalization_value


if __name__ == '__main__':
    class TestDist(BaseDistribution):
        def _func(self, value):
            return tf.exp(-(value - 1.4) ** 2 / 1.8 ** 2)  # non-normalized gaussian


    test1 = TestDist()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(test1.prob(
            tf.cast(np.random.uniform(low=-10., high=10., size=1000000), dtype=tf.float32)))
        result = np.average(result)

        print(sess.run(test1._normalization_value))
    print(result)

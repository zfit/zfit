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
        raise NotImplementedError

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        raise NotImplementedError

    def event_shape_tensor(self, name='event_shape_tensor'):
        raise NotImplementedError


class BaseDistribution(tf.distributions.Distribution, AbstractBaseDistribution):

    def __init__(self, name="BaseDistribution"):
        super(BaseDistribution, self).__init__(dtype=None, reparameterization_type=False,
                                               validate_args=True,
                                               allow_nan_stats=False, name=name)

        self.normalization_opt = {'n_draws': 100000, 'range': (-100., 100.)}
        self._normalization_value = None

    def _func(self, value):
        raise NotImplementedError

    def func(self, value):
        return self._func(value)

    def _prob(self, value):
        pdf = self.func(value) / self.normalization()
        return pdf

    def _normalization_sampler(self):
        lower, upper = self.normalization_opt['range']
        return tf.distributions.Uniform(lower, upper)


    def _set_normalization(self):
        x_sample = self._normalization_sampler().sample(self.normalization_opt['n_draws'])
        avg = tf.reduce_mean(self.func(x_sample))
        self._normalization_value = avg
        return self._normalization_value

    def normalization(self):
        if self._normalization_value is None:
            self._set_normalization()
        return self._normalization_value


if __name__ == '__main__':
    class TestDist(BaseDistribution):
        def _func(self, value):
            return tf.abs(tf.pow(value, 5.) - 3.)


    test1 = TestDist()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(test1.prob(
            tf.cast(np.random.uniform(low=-100., high=100., size=1000000), dtype=tf.float32)))
        result = np.average(result)

    print(result)

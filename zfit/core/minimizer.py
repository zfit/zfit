"""
Definition of minimizers, wrappers etc.

"""

from __future__ import print_function, division, absolute_import

import abc

import tensorflow as tf
import tensorflow_probability as tfp


class AbstractMinimizer(object):
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_variables(self):
        raise NotImplementedError

    @abc.abstractmethod
    def edm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fmin(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tolerance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def status(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gradient(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def func(self):
        raise NotImplementedError


class BaseMinimizer(AbstractMinimizer):

    def __init__(self, func, sess, var_list=None, name="BaseMinimizer", gradients=None):
        self.func = func
        self.sess = sess
        self._start_position = None
        self._gradient = None
        self._variables = None
        self.tolerance = 1e-6
        self.prepare(var_list=var_list)

    @property
    def gradient(self):
        return self._gradient or tf.gradients(self.func(), self.get_variables())[0]

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        self._func = func

    def prepare(self, var_list=None):
        self.set_variables(var_list=var_list)
        self._start_position = tf.convert_to_tensor(
            [v.read_value() for v in self.get_variables() if v.floating()])

    def set_variables(self, var_list=None):
        self._variables = var_list or tf.trainable_variables()

    def get_variables(self):
        return self._variables

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance


class BFGS(BaseMinimizer):

    def __init__(self, *args, **kwargs):
        super(BFGS, self).__init__(*args, **kwargs)

    def minimize(self):
        minimize_fn = tfp.optimizer.bfgs_minimize

        def func(values):
            for param, val in zip(self.get_variables(), tf.unstack(values)):
                param.assign(value=val)
            return self.func(), tf.gradients(self.func(), self.get_variables())[0]

        result = self.sess.run(minimize_fn(func,
                                           initial_position=self._start_position,
                                           tolerance=self.tolerance))

        return result


if __name__ == '__main__':
    from zfit.core.parameter import FitParameter

    with tf.Session() as sess:
        a = FitParameter("blabla", 5.)
        b = FitParameter("blabla2", 55.)
        init = tf.global_variables_initializer()
        sess.run(init)


        def func():
            return (a + b ** 2 + 2) ** 2


        test1 = BFGS(func=func, sess=sess, var_list=[a, b])
        test1.minimize()
        sess.run(test1.minimize())

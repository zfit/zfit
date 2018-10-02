"""
Definition of minimizers, wrappers etc.

"""

from __future__ import print_function, division, absolute_import

import abc

import tensorflow as tf
import tensorflow_probability as tfp

import zfit.core.math as zmath


class AbstractMinimizer(object):
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self):
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


class BaseMinimizer(AbstractMinimizer):

    def __init__(self, name="BaseMinimizer", tolerance=1e-6, sess=None):
        self.sess = sess
        self.tolerance = tolerance

    @property
    def tolerance(self):
        return self._tolerance

    @staticmethod
    def gradient_par(func):
        return zmath.gradient_par(func)

    @staticmethod
    def start_values(parameters):
        """Extract the current value if defined, otherwise random.

        Arguments:
            parameters (FitParameter):

        Return:
            list(const): the current values of parameters
        """
        values = [p.read_value() for p in parameters]
        # TODO: implement if initial val not given
        return values

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance


class BFGS(BaseMinimizer):

    def __init__(self, *args, **kwargs):
        super(BFGS, self).__init__(*args, **kwargs)

    def minimize(self, func, sess=None, gradient=None):
      # with tf.device("/cpu:0"):
        sess = sess or self.sess
        minimizer_fn = tfp.optimizer.bfgs_minimize

        for param in tf.trainable_variables():
            if param.floating():
                print(param)

        params = [p for p in tf.trainable_variables() if p.floating()]

        func_graph = func()

        def to_minimize_func(values):
            # tf.Print(values, [values])
            # print("============values", values)

            # def update_one(param_value):
            #     param, value = param_value
            #     param.update(value=value, session=sess)
            # print("============one param", params[0])
            for param, val in zip(params, tf.unstack(values)):
                param.update(value=val, session=sess)
            # print("DEBUG:", func_graph, tf.gradients(func_graph, params))
            return func_graph, tf.stack(tf.gradients(func_graph, params))

        result = minimizer_fn(to_minimize_func,
                              initial_position=self.start_values(params),
                              tolerance=self.tolerance)

        return result


if __name__ == '__main__':
    from zfit.core.parameter import FitParameter

    with tf.Session() as sess:
        a = FitParameter("blabla", 1.)
        b = FitParameter("blabla2", 2.)
        c = FitParameter("blabla3", 3.1)
        init = tf.global_variables_initializer()
        sess.run(init)


        def func():
            return (a + b ** 2 + 2.*(c-3.)) ** 2


        test1 = BFGS(sess=sess)
        min = test1.minimize(func=func)
        result = sess.run(min)
        print(result)

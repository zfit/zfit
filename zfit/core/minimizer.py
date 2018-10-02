"""
Definition of minimizers, wrappers etc.

"""

from __future__ import print_function, division, absolute_import

import abc

import tensorflow as tf
import tensorflow_probability as tfp

import zfit.core.math as zmath
import zfit.core.tfext as ztf


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

        params = [p for p in tf.trainable_variables() if p.floating()]

        def to_minimize_func(values):
            # tf.Print(values, [values])
            # print("============values", values)

            # def update_one(param_value):
            #     param, value = param_value
            #     param.update(value=value, session=sess)
            # print("============one param", params[0])
            with tf.control_dependencies([values]):
                for param, val in zip(params, tf.unstack(values)):
                    param.update(value=val, session=sess)
                # print("DEBUG:", func_graph, tf.gradients(func_graph, params))
                with tf.control_dependencies([param]):
                    func_graph = func()
            return func_graph, tf.stack(tf.gradients(func_graph, params))

        result = minimizer_fn(to_minimize_func,
                              initial_position=self.start_values(params),
                              tolerance=self.tolerance, parallel_iterations=1)

        return result


if __name__ == '__main__':
    from zfit.core.parameter import FitParameter

    with tf.Session() as sess:
        with tf.variable_scope("func1"):
            a = FitParameter("variable_a", ztf.constant(1.),
                             ztf.constant(-1.),
                             ztf.constant(20.),
                             step_size=ztf.constant(0.001))
        # b = FitParameter("blabla2", 2.)
        # c = FitParameter("blabla3", 3.1)
        init = tf.global_variables_initializer()
        sess.run(init)
        minimizer_fn = tfp.optimizer.bfgs_minimize

        # def func():
        #     return (a + b ** 2 + 2.*(c-3.)) ** 2

        # a = tf.constant(9.0, dtype=tf.float64)
        with tf.control_dependencies([a]):
            def func(a):
                return (a - 1.0) ** 2

        n_steps = 0


        def test_func(val):
            global n_steps
            print("alive!", n_steps)
            global a
            print(a)
            n_steps += 1
            print(val)
            # a = val
            with tf.variable_scope("func1", reuse=True):
                var1 = tf.get_variable(name="variable_a", shape=a.shape, dtype=a.dtype)
            with tf.control_dependencies([val, var1]):
                f = func(var1)
                # a.assign(val, use_locking=True)
                with tf.control_dependencies([var1]):
                    # grad = tf.gradients(f, a)[0]
                    grad = 2. * (var1 - 1.)  # HACK
                    return f, grad


        with tf.control_dependencies([a]):
            min = tfp.optimizer.bfgs_minimize(test_func,
                                              initial_position=tf.constant(10.0, dtype=tf.float64))

        # test1 = BFGS(sess=sess, tolerance=0.001)
        # min = test1.minimize(func=func)
        result = sess.run(min)
        print(result)

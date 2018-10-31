"""
Definition of minimizers, wrappers etc.

"""


import abc
import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.core.math as zmath
from zfit import ztf
import zfit.ztf


class AbstractMinimizer(object):
    """Define the minimizer interface."""

    @abc.abstractmethod
    def minimize(self):
        raise NotImplementedError

    def _minimize(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def edm(self):
        raise NotImplementedError

    def _edm(self, *args, **kwargs):
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


class BaseMinimizer(object):

    def __init__(self, name="BaseMinimizer", tolerance=1e-8, sess=None, *args, **kwargs):
        super(BaseMinimizer, self).__init__(*args, **kwargs)
        self.name = name

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

    def minimize(self, loss, var_list=None):
        return self._call_minimize(loss=loss, var_list=var_list)

    def _call_minimize(self, loss, var_list):
        try:
            return self._minimize(loss=loss, var_list=var_list)
        except NotImplementedError as error:
            try:
                return self._minimize_with_step(loss=loss, var_list=var_list)
            except NotImplementedError:
                raise error

    def _minimize_with_step(self, loss, var_list):  # TODO improve
        changes = collections.deque(np.ones(30))
        last_val = -10
        cur_val = 9999999
        minimum = self.step(loss=loss, var_list=var_list)
        while np.average(changes) > self.tolerance:  # TODO: improve condition
            _ = self.sess.run(minimum)
            changes.popleft()
            changes.append(abs(cur_val - last_val))
            last_val = cur_val
            cur_val = self.sess.run(loss)
        self.sess.run([v for v in var_list])
        return cur_val


# WIP
# class BFGS(BaseMinimizer):
#
#     def __init__(self, *args, **kwargs):
#         super(BFGS, self).__init__(*args, **kwargs)
#
#     def minimize(self, func, sess=None, gradient=None):
#         # with tf.device("/cpu:0"):
#         sess = sess or self.sess
#         minimizer_fn = tfp.optimizer.bfgs_minimize
#
#         params = [p for p in tf.trainable_variables() if p.floating()]
#
#         def to_minimize_func(values):
#             # tf.Print(values, [values])
#             # print("============values", values)
#
#             # def update_one(param_value):
#             #     param, value = param_value
#             #     param.update(value=value, session=sess)
#             # print("============one param", params[0])
#             with tf.control_dependencies([values]):
#                 for param, val in zip(params, tf.unstack(values)):
#                     param.update(value=val, session=sess)
#                 with tf.control_dependencies([param]):
#                     func_graph = func()
#             return func_graph, tf.stack(tf.gradients(func_graph, params))
#
#         result = minimizer_fn(to_minimize_func,
#                               initial_position=self.start_values(params),
#                               tolerance=self.tolerance, parallel_iterations=1)
#
#         return result


# TensorFlow Minimizer

class HelperAdapterTFOptimizer(object):
    """Adapter for tf.Optimizer to convert the step-by-step minimization to full minimization"""

    def __init__(self, *args, **kwargs):  # self check
        assert issubclass(self.__class__, tf.train.Optimizer)  # assumption
        super(HelperAdapterTFOptimizer, self).__init__(*args, **kwargs)

    def step(self, loss, var_list):
        """One step of the minimization. Equals to `tf.train.Optimizer.minimize`

        Args:
            loss (graph): The loss function to minimize
            var_list (list(tf.Variable...)): A list of tf.Variables that will be optimized.


        """
        minimum = super(HelperAdapterTFOptimizer, self).minimize(loss=loss, var_list=var_list)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        return minimum


class AdapterTFOptimizer(BaseMinimizer, HelperAdapterTFOptimizer):
    pass


# Explicit classes to use
class AdadeltaMinimizer(AdapterTFOptimizer, tf.train.AdadeltaOptimizer, AbstractMinimizer):
    pass


class AdagradMinimizer(AdapterTFOptimizer, tf.train.AdagradOptimizer, AbstractMinimizer):
    pass


class GradientDescentMinimizer(AdapterTFOptimizer, tf.train.GradientDescentOptimizer,
                               AbstractMinimizer):
    pass


class RMSPropMinimizer(AdapterTFOptimizer, tf.train.RMSPropOptimizer, AbstractMinimizer):
    pass


class AdamMinimizer(AdapterTFOptimizer, tf.train.AdamOptimizer, AbstractMinimizer):
    pass


# WIP below
if __name__ == '__main__':
    from zfit.core.parameter import FitParameter

    with tf.Session() as sess:
        with tf.variable_scope("func1"):
            a = FitParameter("variable_a", ztf.constant(1.5),
                             ztf.constant(-1.),
                             ztf.constant(20.),
                             step_size=ztf.constant(0.1))
            b = FitParameter("variable_b", 2.)
            c = FitParameter("variable_c", 3.1)
        minimizer_fn = tfp.optimizer.bfgs_minimize


        def func():
            return (a - 1.) ** 2 + (b - 4.) ** 2 + (c - 8) ** 4


        # a = tf.constant(9.0, dtype=tf.float64)
        # with tf.control_dependencies([a]):
        #     def func(a):
        #         return (a - 1.0) ** 2

        n_steps = 0

        #
        # def test_func(val):
        #     global n_steps
        #     print("alive!", n_steps)
        #     global a
        #     print(a)
        #     n_steps += 1
        #     print(val)
        #     # a = val
        #     with tf.variable_scope("func1", reuse=True):
        #         var1 = tf.get_variable(name="variable_a", shape=a.shape, dtype=a.dtype)
        #     with tf.control_dependencies([val, var1]):
        #         f = func(var1)
        #         # a.assign(val, use_locking=True)
        #         with tf.control_dependencies([var1]):
        #             # grad = tf.gradients(f, a)[0]
        #             grad = 2. * (var1 - 1.)  # HACK
        #             return f, grad

        loss_func = func()
        # with tf.control_dependencies([a]):
        #     min = tfp.optimizer.bfgs_minimize(test_func,
        #                                       initial_position=tf.constant(10.0,
        # dtype=tf.float64))
        # minimizer = tf.train.AdamOptimizer()

        # min = minimizer.minimize(loss=loss_func, var_list=[a, b, c])
        minimizer = AdamMinimizer(sess=sess, learning_rate=0.3)

        # test1 = BFGS(sess=sess, tolerance=0.001)
        # min = test1.minimize(func=func)
        # last_val = 100000
        # cur_val = 9999999
        # while abs(last_val - cur_val) > 0.0000000000000000000001:
        #     result = sess.run(min)
        #     last_val = cur_val
        #     cur_val = sess.run(loss_func)
        #     print("running")
        # aval, bval, cval = sess.run([v for v in (a, b, c)])
        # aval, bval, cval = sess.run([v.read_value() for v in (a, b, c)])
        # print("a, b, c", aval, bval, cval)
        minimizer.minimize(loss=loss_func, var_list=[a, b, c])
        cur_val = sess.run(loss_func)
        result = cur_val
        print(sess.run([v.read_value() for v in (a, b, c)]))
        print(result)

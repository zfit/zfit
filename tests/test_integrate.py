from __future__ import print_function, division, absolute_import

from unittest import TestCase

import pytest
import tensorflow as tf

import zfit.core.integrate as zintegrate

limits1_5deps = [(1., -1., 2., 4., 3.), (5., 4., 5., 8., 9.) ]
limits_simple_5deps = (1., 5.)
limits_simple_5deps = [(1., 1., 1., 1., 1.), (5., 5., 5., 5., 5.)]


def func1_5deps(value):
    a, b, c, d, e = tf.unstack(value)
    return a + b * c ** 2 + d ** 2 * e ** 3


def func1_5deps_fully_integrated(limits):
    lower, upper = limits

    def def_int(x):
        a, b, c, d, e = x
        val = 0.5 * a ** 2 + 0.5 * b ** 2 * (1. / 3.) * c ** 3 + (1. / 3.) * d ** 3 * (0.25) * e ** 4
        return val

    return def_int(upper) - def_int(lower)


# @pytest.mark.parametrize
def test_mc_integration():
    num_integral = zintegrate.mc_integrate(func=func1_5deps, limits=limits_simple_5deps, n_dims=5,
                                           draws_per_dim=8)
    with tf.Session() as sess:

        integral = sess.run(num_integral)
        assert len(integral) == 1
        assert func1_5deps_fully_integrated(limits_simple_5deps) == integral

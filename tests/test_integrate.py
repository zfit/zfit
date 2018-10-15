from __future__ import print_function, division, absolute_import

from unittest import TestCase

import pytest
import tensorflow as tf
import numpy as np

import zfit.core.integrate as zintegrate

limits1_5deps = [(1., -1., 2., 4., 3.), (5., 4., 5., 8., 9.)]
limits_simple_5deps = (1., 5.)
limits_simple_5deps = [(1., 1., 1., 1., 1.), (5., 5., 5., 5., 5.)]


def func1_5deps(value):
    a, b, c, d, e = tf.unstack(value)
    return a + b * c ** 2 + d ** 2 * e ** 3


def func1_5deps_fully_integrated(limits):
    lower, upper = limits
    a_lower, b_lower, c_lower, d_lower, e_lower = lower
    a_upper, b_upper, c_upper, d_upper, e_upper = upper

    val = -e_lower ** 4 * (
    a_lower * b_lower * c_lower * d_lower ** 3 / 12 - a_lower * b_lower * c_lower *
    d_upper ** 3 / 12 - a_lower * b_lower * c_upper * d_lower ** 3 / 12 + a_lower *
    b_lower * c_upper * d_upper ** 3 / 12 - a_lower * b_upper * c_lower * d_lower **
    3 / 12 + a_lower * b_upper * c_lower * d_upper ** 3 / 12 + a_lower * b_upper *
    c_upper * d_lower ** 3 / 12 - a_lower * b_upper * c_upper * d_upper ** 3 / 12 -
    a_upper * b_lower * c_lower * d_lower ** 3 / 12 + a_upper * b_lower * c_lower *
    d_upper ** 3 / 12 + a_upper * b_lower * c_upper * d_lower ** 3 / 12 - a_upper *
    b_lower * c_upper * d_upper ** 3 / 12 + a_upper * b_upper * c_lower * d_lower **
    3 / 12 - a_upper * b_upper * c_lower * d_upper ** 3 / 12 - a_upper * b_upper *
    c_upper * d_lower ** 3 / 12 + a_upper * b_upper * c_upper * d_upper ** 3 / 12) - \
    e_lower * (
    a_lower ** 2 * b_lower * c_lower * d_lower / 2 - a_lower ** 2 * b_lower *
    c_lower * d_upper / 2 - a_lower ** 2 * b_lower * c_upper * d_lower / 2 +
    a_lower ** 2 * b_lower * c_upper * d_upper / 2 - a_lower ** 2 * b_upper *
    c_lower * d_lower / 2 + a_lower ** 2 * b_upper * c_lower * d_upper / 2 +
    a_lower ** 2 * b_upper * c_upper * d_lower / 2 - a_lower ** 2 * b_upper *
    c_upper * d_upper / 2 + a_lower * b_lower ** 2 * c_lower ** 3 * d_lower / 6
    - a_lower * b_lower ** 2 * c_lower ** 3 * d_upper / 6 - a_lower * b_lower
    ** 2 * c_upper ** 3 * d_lower / 6 + a_lower * b_lower ** 2 * c_upper ** 3 *
    d_upper / 6 - a_lower * b_upper ** 2 * c_lower ** 3 * d_lower / 6 + a_lower
    * b_upper ** 2 * c_lower ** 3 * d_upper / 6 + a_lower * b_upper ** 2 *
    c_upper ** 3 * d_lower / 6 - a_lower * b_upper ** 2 * c_upper ** 3 *
    d_upper / 6 - a_upper ** 2 * b_lower * c_lower * d_lower / 2 + a_upper ** 2
    * b_lower * c_lower * d_upper / 2 + a_upper ** 2 * b_lower * c_upper *
    d_lower / 2 - a_upper ** 2 * b_lower * c_upper * d_upper / 2 + a_upper ** 2
    * b_upper * c_lower * d_lower / 2 - a_upper ** 2 * b_upper * c_lower *
    d_upper / 2 - a_upper ** 2 * b_upper * c_upper * d_lower / 2 + a_upper ** 2
    * b_upper * c_upper * d_upper / 2 - a_upper * b_lower ** 2 * c_lower ** 3 *
    d_lower / 6 + a_upper * b_lower ** 2 * c_lower ** 3 * d_upper / 6 + a_upper
    * b_lower ** 2 * c_upper ** 3 * d_lower / 6 - a_upper * b_lower ** 2 *
    c_upper ** 3 * d_upper / 6 + a_upper * b_upper ** 2 * c_lower ** 3 *
    d_lower / 6 - a_upper * b_upper ** 2 * c_lower ** 3 * d_upper / 6 - a_upper
    * b_upper ** 2 * c_upper ** 3 * d_lower / 6 + a_upper * b_upper ** 2 *
    c_upper ** 3 * d_upper / 6) + e_upper ** 4 * (
    a_lower * b_lower * c_lower * d_lower ** 3 / 12 - a_lower * b_lower *
    c_lower * d_upper ** 3 / 12 - a_lower * b_lower * c_upper * d_lower ** 3 /
    12 + a_lower * b_lower * c_upper * d_upper ** 3 / 12 - a_lower * b_upper *
    c_lower * d_lower ** 3 / 12 + a_lower * b_upper * c_lower * d_upper ** 3 /
    12 + a_lower * b_upper * c_upper * d_lower ** 3 / 12 - a_lower * b_upper *
    c_upper * d_upper ** 3 / 12 - a_upper * b_lower * c_lower * d_lower ** 3 /
    12 + a_upper * b_lower * c_lower * d_upper ** 3 / 12 + a_upper * b_lower *
    c_upper * d_lower ** 3 / 12 - a_upper * b_lower * c_upper * d_upper ** 3 /
    12 + a_upper * b_upper * c_lower * d_lower ** 3 / 12 - a_upper * b_upper *
    c_lower * d_upper ** 3 / 12 - a_upper * b_upper * c_upper * d_lower ** 3 /
    12 + a_upper * b_upper * c_upper * d_upper ** 3 / 12) + e_upper * (
    a_lower ** 2 * b_lower * c_lower * d_lower / 2 - a_lower ** 2 * b_lower *
    c_lower * d_upper / 2 - a_lower ** 2 * b_lower * c_upper * d_lower / 2 +
    a_lower ** 2 * b_lower * c_upper * d_upper / 2 - a_lower ** 2 * b_upper *
    c_lower * d_lower / 2 + a_lower ** 2 * b_upper * c_lower * d_upper / 2 +
    a_lower ** 2 * b_upper * c_upper * d_lower / 2 - a_lower ** 2 * b_upper *
    c_upper * d_upper / 2 + a_lower * b_lower ** 2 * c_lower ** 3 * d_lower / 6
    - a_lower * b_lower ** 2 * c_lower ** 3 * d_upper / 6 - a_lower * b_lower
    ** 2 * c_upper ** 3 * d_lower / 6 + a_lower * b_lower ** 2 * c_upper ** 3 *
    d_upper / 6 - a_lower * b_upper ** 2 * c_lower ** 3 * d_lower / 6 + a_lower *
    b_upper ** 2 * c_lower ** 3 * d_upper / 6 + a_lower * b_upper ** 2 * c_upper **
    3 * d_lower / 6 - a_lower * b_upper ** 2 * c_upper ** 3 * d_upper / 6 - a_upper
    ** 2 * b_lower * c_lower * d_lower / 2 + a_upper ** 2 * b_lower * c_lower *
    d_upper / 2 + a_upper ** 2 * b_lower * c_upper * d_lower / 2 - a_upper ** 2 *
    b_lower * c_upper * d_upper / 2 + a_upper ** 2 * b_upper * c_lower * d_lower /
    2 - a_upper ** 2 * b_upper * c_lower * d_upper / 2 - a_upper ** 2 * b_upper *
    c_upper * d_lower / 2 + a_upper ** 2 * b_upper * c_upper * d_upper / 2 -
    a_upper * b_lower ** 2 * c_lower ** 3 * d_lower / 6 + a_upper * b_lower ** 2 *
    c_lower ** 3 * d_upper / 6 + a_upper * b_lower ** 2 * c_upper ** 3 * d_lower /
    6 - a_upper * b_lower ** 2 * c_upper ** 3 * d_upper / 6 + a_upper * b_upper **
    2 * c_lower ** 3 * d_lower / 6 - a_upper * b_upper ** 2 * c_lower ** 3 *
    d_upper / 6 - a_upper * b_upper ** 2 * c_upper ** 3 * d_lower / 6 + a_upper *
    b_upper ** 2 * c_upper ** 3 * d_upper / 6)
    return val




limits2 = (-1., 2.)


def func2_1deps(value):
    a = value
    return a ** 2


def func2_1deps_fully_integrated(limits):
    lower, upper = limits

    def func_int(x):
        return (1 / 3) * x ** 3

    return func_int(upper) - func_int(lower)


limits3 = [(-1., -1.), (2., 2.)]


def func3_2deps(value):
    a, b = tf.unstack(value)
    return a ** 2 + b ** 2


def func3_2deps_fully_integrated(limits):
    lower, upper = limits
    lower_a, lower_b = lower
    upper_a, upper_b = upper
    integral = (lower_a ** 3 - upper_a ** 3) * (lower_b - upper_b)
    integral += (lower_a - upper_a) * (lower_b ** 3 - upper_b ** 3)
    integral /= 3
    return integral


# @pytest.mark.parametrize


def test_mc_integration():
    # simpel example
    num_integral = zintegrate.mc_integrate(func=func1_5deps, limits=limits_simple_5deps, n_dims=5,
                                           draws_per_dim=8)
    num_integral2 = zintegrate.mc_integrate(func=func2_1deps, limits=limits2, n_dims=1)
    num_integral3 = zintegrate.mc_integrate(func=func3_2deps, limits=limits3, n_dims=2,
                                            draws_per_dim=1000)

    with tf.Session() as sess:
        integral = sess.run(num_integral)
        integral2 = sess.run(num_integral2)
        integral3 = sess.run(num_integral3)

        assert not hasattr(integral, "__len__")
        assert not hasattr(integral2, "__len__")
        assert not hasattr(integral3, "__len__")
        assert func1_5deps_fully_integrated(limits_simple_5deps) == pytest.approx(integral,
                                                                                  rel=0.05)
        assert func2_1deps_fully_integrated(limits2) == pytest.approx(integral2, rel=0.03)
        assert func3_2deps_fully_integrated(limits3) == pytest.approx(integral3, rel=0.03)

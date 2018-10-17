from __future__ import print_function, division, absolute_import

import tensorflow as tf

import zfit

import zfit.core.minimizer as zmin
import zfit.core.tfext as ztf


def minimize_func(minimizer_class):
    from zfit.core.parameter import FitParameter

    parameter_tolerance = 0.5
    max_distance_to_min = 0.5

    with tf.Session() as sess:
        with tf.variable_scope("func1"):
            true_a = 1.
            true_b = 4.
            true_c = 7.
            a = FitParameter("variable_a", ztf.constant(1.5),
                             ztf.constant(-1.),
                             ztf.constant(20.),
                             step_size=ztf.constant(0.1))
            b = FitParameter("variable_b", 3.5)
            c = FitParameter("variable_c", 7.8)

        def func():
            return (a - true_a) ** 2 + (b - true_b) ** 2 + (c - true_c) ** 4
        loss_func = func()
        minimizer = minimizer_class(sess=sess, learning_rate=0.3, tolerance=0.0002)

        minimizer.minimize(loss=loss_func, var_list=[a, b, c])
        cur_val = sess.run(loss_func)
        aval, bval, cval = sess.run([v.read_value() for v in (a, b, c)])
    assert cur_val < max_distance_to_min
    assert abs(aval - true_a) < parameter_tolerance
    assert abs(bval - true_b) < parameter_tolerance
    assert abs(cval - true_c) < parameter_tolerance



minimizers = [zmin.AdamMinimizer,
              zmin.AdadeltaMinimizer,
              zmin.AdagradMinimizer,
              zmin.GradientDescentMinimizer,
              zmin.RMSPropMinimizer]

def test_minimizers():
    for minimizer_class in minimizers:
        minimize_func(minimizer_class)

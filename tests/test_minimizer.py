import pytest
import tensorflow as tf

from zfit.core.loss import SimpleLoss
import zfit.core.minimizer as zmin
from zfit import ztf
import zfit.minimizers.optimizers_tf


def minimize_func(minimizer_class_and_kwargs, sess):
    from zfit.core.parameter import FitParameter

    parameter_tolerance = 0.3
    max_distance_to_min = 0.11

    with tf.variable_scope("func1"):
        true_a = 1.
        true_b = 4.
        true_c = 7.
        a_param = FitParameter("variable_a", ztf.constant(1.5),
                               ztf.constant(-1.),
                               ztf.constant(20.),
                               step_size=ztf.constant(0.1))
        b_param = FitParameter("variable_b", 3.5)
        c_param = FitParameter("variable_c", 7.8)

    def func(a, b, c):
        return tf.convert_to_tensor((a - true_a) ** 6 + (b - true_b) ** 2 + (c - true_c) ** 4) + 0.42

    # print("DEBUG": before true_minimum")

    true_minimum = sess.run(func(true_a, true_b, true_c))
    # print("DEBUG": true_minimum", true_minimum)
    loss_func_tf = func(a_param, b_param, c_param)

    def loss_to_call():
        return loss_func_tf

    loss_func = SimpleLoss(loss_to_call)

    minimizer_class, minimizer_kwargs = minimizer_class_and_kwargs
    minimizer = minimizer_class(loss=loss_func, **minimizer_kwargs)
    init = tf.initialize_all_variables()
    sess.run(init)

    minimizer.minimize(sess=sess, params=[a_param, b_param, c_param])
    cur_val = sess.run(loss_func.eval())
    aval, bval, cval = sess.run([v.read_value() for v in (a_param, b_param, c_param)])

    assert abs(cur_val - true_minimum) < max_distance_to_min
    assert abs(aval - true_a) < parameter_tolerance
    assert abs(bval - true_b) < parameter_tolerance
    assert abs(cval - true_c) < parameter_tolerance


minimizers = [(zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.train.AdamOptimizer())),
              (zfit.minimizers.optimizers_tf.AdamMinimizer, dict(learning_rate=0.3)),
              # zmin.AdadeltaMinimizer,  # not working well...
              # (zfit.minimizers.optimizers_tf.AdagradMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
              # (zfit.minimizers.optimizers_tf.GradientDescentMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
              # (zfit.minimizers.optimizers_tf.RMSPropMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
              # (zfit.minimize.MinuitTFMinimizer, {}),
              (zfit.minimize.MinuitMinimizer, {}),
              (zfit.minimize.ScipyMinimizer, {}),
              ]


# print("DEBUG": after minimizer instanciation")

@pytest.mark.parametrize("minimizer_class", minimizers)
def test_minimizers(minimizer_class):
    # for minimizer_class in minimizers:
    with tf.Session() as sess:
        minimize_func(minimizer_class, sess=sess)


if __name__ == '__main__':
    with tf.Session() as sess:
        for minimizer in minimizers:
            test_minimizers(minimizer_class=minimizer)

import pytest
import tensorflow as tf
import numpy as np

from zfit.core.loss import SimpleLoss
import zfit.minimizers.baseminimizer as zmin
from zfit import ztf
import zfit.minimizers.optimizers_tf


def minimize_func(minimizer_class_and_kwargs):
    from zfit.core.parameter import Parameter

    parameter_tolerance = 0.05
    max_distance_to_min = 0.8

    with tf.variable_scope("func1"):
        true_a = 1.
        a_data = np.random.normal(loc=true_a, size=10000, scale=0.1)
        true_b = 4.
        true_c = 7.
        a_param = Parameter("variable_a", ztf.constant(1.5),
                            ztf.constant(-1.),
                            ztf.constant(20.),
                            step_size=ztf.constant(0.1))
        b_param = Parameter("variable_b", 3.5)
        c_param = Parameter("variable_c", 7.8)

    def func(a, b, c):
        probs = ztf.convert_to_tensor((a - a_data) ** 2 + (b - true_b) ** 2 + (c - true_c) ** 4) + 0.42
        # return tf.reduce_sum(tf.log(probs))
        return tf.reduce_sum(tf.log(probs))

    true_minimum = zfit.run(func(true_a, true_b, true_c))
    # print("DEBUG: true_minimum", true_minimum)
    loss_func_tf = func(a_param, b_param, c_param)

    def loss_to_call():
        return loss_func_tf

    loss_func = SimpleLoss(loss_to_call)

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)

    result = minimizer.minimize(loss=loss_func, params=[a_param, b_param, c_param])
    cur_val = zfit.run(loss_func.value())
    aval, bval, cval = zfit.run([v for v in (a_param, b_param, c_param)])

    assert abs(aval - true_a) < parameter_tolerance
    assert abs(bval - true_b) < parameter_tolerance
    assert abs(cval - true_c) < parameter_tolerance
    assert abs(cur_val - true_minimum) < max_distance_to_min

    if test_error:
        # Test Error
        a_errors = result.error(params=a_param)
        assert tuple(a_errors.keys()) == (a_param,)
        errors = result.error()
        a_error = a_errors[a_param]
        assert a_error['lower'] == pytest.approx(-a_error['upper'], rel=0.01)
        assert abs(a_error['lower']) == pytest.approx(0.0067, rel=0.07)
        assert abs(errors[b_param]['lower']) == pytest.approx(0.0067, rel=0.07)
        assert abs(errors[c_param]['lower']) == pytest.approx(0.074, rel=0.07)
        assert abs(errors[c_param]['upper']) == pytest.approx(0.088, rel=0.07)

        assert errors[a_param]['lower'] == pytest.approx(a_error['lower'], rel=0.01)
        assert errors[a_param]['upper'] == pytest.approx(a_error['upper'], rel=0.01)

        # Test Hesse
        b_hesses = result.hesse(params=b_param)
        assert tuple(b_hesses.keys()) == (b_param,)
        errors = result.hesse()
        b_hesse = b_hesses[b_param]
        assert abs(b_hesse['error']) == pytest.approx(0.0065, rel=0.07)
        assert abs(errors[b_param]['error']) == pytest.approx(0.0065, rel=0.07)
        assert abs(errors[c_param]['error']) == pytest.approx(0.3, rel=0.07)

        assert errors[b_param]['error'] == pytest.approx(b_hesse['error'], rel=0.01)


minimizers = [
    (zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.train.AdamOptimizer(learning_rate=0.5)), False),
    (zfit.minimizers.optimizers_tf.AdamMinimizer, dict(learning_rate=0.5), False),
    # zmin.AdadeltaMinimizer,  # not working well...
    # (zfit.minimizers.optimizers_tf.AdagradMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
    # (zfit.minimizers.optimizers_tf.GradientDescentMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
    # (zfit.minimizers.optimizers_tf.RMSPropMinimizer, dict(learning_rate=0.4, tolerance=0.3)),
    # (zfit.minimize.MinuitTFMinimizer, {}),
    (zfit.minimize.MinuitMinimizer, {}, True),
    (zfit.minimize.ScipyMinimizer, {}, False),
    ]


# print("DEBUG": after minimizer instantiation")

@pytest.mark.parametrize("minimizer_class", minimizers)
def test_minimizers(minimizer_class):
    # for minimizer_class in minimizers:
    minimize_func(minimizer_class)


if __name__ == '__main__':
    for minimizer in minimizers:
        test_minimizers(minimizer_class=minimizer)

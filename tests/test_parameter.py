import numpy as np
import pytest

import tensorflow as tf

import zfit
from zfit import Parameter, ztf
from zfit.core.parameter import ComposedParameter, ComplexParameter
from zfit.util.exception import LogicalUndefinedOperationError


def test_complex_param():
    real_part = 1.3
    imag_part = 0.3
    complex_value = real_part + imag_part * 1.j
    param1 = ComplexParameter("param1", complex_value)
    some_value = 3. * param1 ** 2 - 1.2j
    true_value = 3. * complex_value ** 2 - 1.2j
    zfit.sess.run(tf.global_variables_initializer())
    assert true_value == pytest.approx(zfit.sess.run(some_value), rel=1e-8)
    part1, part2 = param1.get_dependents()
    part1_val, part2_val = zfit.sess.run([part1.value(), part2.value()])
    if part1_val == pytest.approx(real_part):
        assert part2_val == pytest.approx(imag_part)
    elif part2_val == pytest.approx(real_part):
        assert part1_val == pytest.approx(imag_part)
    else:
        assert False, "one of the if or elif should be the case"


def test_composed_param():
    # tf.reset_default_graph()
    param1 = Parameter('param1s', 1.)
    param2 = Parameter('param2s', 2.)
    param3 = Parameter('param3s', 3., floating=False)
    param4 = Parameter('param4s', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    param_a = ComposedParameter('param_as', tensor=a)
    assert isinstance(param_a.get_dependents(only_floating=True), set)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}
    zfit.sess.run(tf.global_variables_initializer())
    a_unchanged = zfit.sess.run(a)
    assert a_unchanged == zfit.sess.run(param_a)
    assert zfit.sess.run(param2.assign(3.5))
    a_changed = zfit.sess.run(a)
    assert a_changed == zfit.sess.run(param_a)
    assert a_changed != a_unchanged

    with pytest.raises(LogicalUndefinedOperationError):
        param_a.assign(value=5.)
    with pytest.raises(LogicalUndefinedOperationError):
        param_a.load(value=5., session=zfit.sess)


def test_param_limits():
    lower, upper = -4., 3.
    param1 = Parameter('param1lim', 1., lower_limit=lower, upper_limit=upper)
    param2 = Parameter('param2lim', 2.)

    zfit.sess.run(tf.global_variables_initializer())
    param1.load(upper + 0.5, session=zfit.sess)
    assert upper == zfit.sess.run(param1.value())
    param1.load(lower - 1.1, session=zfit.sess)
    assert lower == zfit.sess.run(param1.value())
    param2.lower_limit = lower
    param2.load(lower - 1.1, session=zfit.sess)
    assert lower == zfit.sess.run(param2.value())

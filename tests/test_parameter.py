from math import pi, cos

import numpy as np
import pytest

import tensorflow as tf

import zfit
from zfit import Parameter, ztf
from zfit.core.parameter import ComposedParameter, ComplexParameter
from zfit.util.exception import LogicalUndefinedOperationError, NameAlreadyTakenError
from zfit.core.testing import setup_function, teardown_function, tester


def test_complex_param():
    real_part = 1.3
    imag_part = 0.3
    # Constant complex
    complex_value = real_part + imag_part * 1.j
    param1 = ComplexParameter("param1_compl", complex_value)
    some_value = 3. * param1 ** 2 - 1.2j
    true_value = 3. * complex_value ** 2 - 1.2j
    assert true_value == pytest.approx(zfit.run(some_value), rel=1e-8)
    assert not param1.get_dependents()
    # Cartesian complex
    real_part_param = Parameter("real_part_param", real_part)
    imag_part_param = Parameter("imag_part_param", imag_part)
    param2 = ComplexParameter.from_cartesian("param2_compl", real_part_param, imag_part_param)
    part1, part2 = param2.get_dependents()
    part1_val, part2_val = zfit.run([part1.value(), part2.value()])
    if part1_val == pytest.approx(real_part):
        assert part2_val == pytest.approx(imag_part)
    elif part2_val == pytest.approx(real_part):
        assert part1_val == pytest.approx(imag_part)
    else:
        assert False, "one of the if or elif should be the case"
    # Polar complex
    mod_val = 1.0
    arg_val = pi / 4.0
    mod_part_param = Parameter("mod_part_param", mod_val)
    arg_part_param = Parameter("arg_part_param", arg_val)
    param3 = ComplexParameter.from_polar("param3_compl", mod_part_param, arg_part_param)
    part1, part2 = param3.get_dependents()
    part1_val, part2_val = zfit.run([part1.value(), part2.value()])
    if part1_val == pytest.approx(mod_val):
        assert part2_val == pytest.approx(arg_val)
    elif part1_val == pytest.approx(arg_val):
        assert part2_val == pytest.approx(mod_val)
    else:
        assert False, "one of the if or elif should be the case"

    param4_name = "param4"
    param4 = ComplexParameter.from_polar(param4_name, 4., 2., floating=True)
    deps_param4 = param4.get_dependents()
    assert len(deps_param4) == 2
    for dep in deps_param4:
        assert dep.floating
    assert param4.mod.name == param4_name + "_mod"
    assert param4.arg.name == param4_name + "_arg"

    # Test properties (1e-8 is too precise)
    assert real_part == pytest.approx(zfit.run(param1.real), rel=1e-6)
    assert imag_part == pytest.approx(zfit.run(param2.imag), rel=1e-6)
    assert mod_val == pytest.approx(zfit.run(param3.mod), rel=1e-6)
    assert arg_val == pytest.approx(zfit.run(param3.arg), rel=1e-6)
    assert cos(arg_val) == pytest.approx(zfit.run(param3.real), rel=1e-6)


def test_composed_param():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    param_a = ComposedParameter('param_as', tensor=a)
    assert isinstance(param_a.get_dependents(only_floating=True), set)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}
    a_unchanged = zfit.run(a)
    assert a_unchanged == zfit.run(param_a)
    assert zfit.run(param2.assign(3.5))
    a_changed = zfit.run(a)
    assert a_changed == zfit.run(param_a)
    assert a_changed != a_unchanged

    with pytest.raises(LogicalUndefinedOperationError):
        param_a.assign(value=5.)
    with pytest.raises(LogicalUndefinedOperationError):
        param_a.load(value=5., session=zfit.run.sess)


def test_param_limits():
    lower, upper = -4., 3.
    param1 = Parameter('param1', 1., lower_limit=lower, upper_limit=upper)
    param2 = Parameter('param2', 2.)

    param1.load(upper + 0.5)
    assert upper == zfit.run(param1.value())
    param1.load(lower - 1.1)
    assert lower == zfit.run(param1.value())
    param2.lower_limit = lower
    param2.load(lower - 1.1)
    assert lower == zfit.run(param2.value())


def test_overloaded_operators():
    param_a = ComposedParameter('param_a', 5 * 4)
    param_b = ComposedParameter('param_b', 3)
    param_c = param_a * param_b
    assert not isinstance(param_c, zfit.Parameter)
    param_d = ComposedParameter("param_d", param_a + param_a * param_b ** 2)
    param_d_val = zfit.run(param_d)
    assert param_d_val == zfit.run(param_a + param_a * param_b ** 2)


def test_equal_naming():
    param_unique_name = zfit.Parameter('fafdsfds', 5.)
    with pytest.raises(NameAlreadyTakenError):
        param_unique_name2 = zfit.Parameter('fafdsfds', 3.)


def test_set_value():
    value1 = 1.
    value2 = 2.
    value3 = 3.
    value4 = 4.
    param1 = zfit.Parameter(name="test_set_value15", value=value1)
    assert zfit.run(param1) == value1
    with param1.set_value(value2):
        assert zfit.run(param1) == value2
        param1.set_value(value3)
        assert zfit.run(param1) == value3
        with param1.set_value(value4):
            assert zfit.run(param1) == value4
        assert zfit.run(param1) == value3
    assert zfit.run(param1) == value1

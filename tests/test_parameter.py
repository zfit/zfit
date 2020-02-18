#  Copyright (c) 2020 zfit
from math import pi, cos

import pytest
import tensorflow as tf
from ordered_set import OrderedSet

import zfit
from zfit import Parameter, z
from zfit.core.parameter import ComposedParameter, ComplexParameter
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.exception import LogicalUndefinedOperationError, NameAlreadyTakenError


def test_complex_param():
    real_part = 1.3
    imag_part = 0.3

    # Constant complex
    def complex_value():
        return real_part + imag_part * 1.j

    param1 = ComplexParameter("param1_compl", complex_value, dependents=None)
    some_value = 3. * param1 ** 2 - 1.2j
    true_value = 3. * complex_value() ** 2 - 1.2j
    assert true_value == pytest.approx(some_value.numpy(), rel=1e-8)
    assert not param1.get_dependents()
    # Cartesian complex
    real_part_param = Parameter("real_part_param", real_part)
    imag_part_param = Parameter("imag_part_param", imag_part)
    param2 = ComplexParameter.from_cartesian("param2_compl", real_part_param, imag_part_param)
    part1, part2 = param2.get_dependents()
    part1_val, part2_val = [part1.value().numpy(), part2.value().numpy()]
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
    part1_val, part2_val = [part1.value().numpy(), part2.value().numpy()]
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
    assert real_part == pytest.approx(param1.real.numpy(), rel=1e-6)
    assert imag_part == pytest.approx(param2.imag.numpy(), rel=1e-6)
    assert mod_val == pytest.approx(param3.mod.numpy(), rel=1e-6)
    assert arg_val == pytest.approx(param3.arg.numpy(), rel=1e-6)
    assert cos(arg_val) == pytest.approx(param3.real.numpy(), rel=1e-6)


def test_composed_param():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)  # needed to make sure it does not only take all params as deps

    def a():
        return z.log(3. * param1) * tf.square(param2) - param3

    param_a = ComposedParameter('param_as', value_fn=a, dependents=(param1, param2, param3))
    assert isinstance(param_a.get_dependents(only_floating=True), OrderedSet)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}
    a_unchanged = a().numpy()
    assert a_unchanged == param_a.numpy()
    assert param2.assign(3.5).numpy()
    a_changed = a().numpy()
    assert a_changed == param_a.numpy()
    assert a_changed != a_unchanged

    with pytest.raises(LogicalUndefinedOperationError):
        param_a.assign(value=5.)
    with pytest.raises(LogicalUndefinedOperationError):
        param_a.assign(value=5.)


def test_floating_behavior():
    param1 = zfit.Parameter('param1', 1.0)
    assert param1.floating


def test_param_limits():
    lower, upper = -4., 3.
    param1 = Parameter('param1', 1., lower_limit=lower, upper_limit=upper)
    param2 = Parameter('param2', 2.)

    param1.load(upper + 0.5)
    assert upper == param1.value().numpy()
    param1.load(lower - 1.1)
    assert lower == param1.value().numpy()
    param2.lower_limit = lower
    param2.load(lower - 1.1)
    assert lower == param2.value().numpy()


def test_overloaded_operators():
    param_a = ComposedParameter('param_a', lambda: 5 * 4, dependents=None)
    param_b = ComposedParameter('param_b', lambda: 3, dependents=None)
    param_c = param_a * param_b
    assert not isinstance(param_c, zfit.Parameter)
    param_d = ComposedParameter("param_d", lambda: param_a + param_a * param_b ** 2, dependents=[param_a, param_b])
    param_d_val = param_d.numpy()
    assert param_d_val == (param_a + param_a * param_b ** 2).numpy()


# @pytest.mark.skip  # TODO: reactivate, causes segfault
def test_equal_naming():
    param_unique_name = zfit.Parameter('fafdsfds', 5.)
    with pytest.raises(NameAlreadyTakenError):
        param_unique_name2 = zfit.Parameter('fafdsfds', 3.)


# @pytest.mark.skip  # TODO: segfaulting?
def test_set_value():
    value1 = 1.
    value2 = 2.
    value3 = 3.
    value4 = 4.
    param1 = zfit.Parameter(name="test_set_value15", value=value1)
    assert param1.numpy() == value1
    with param1.set_value(value2):
        assert param1.numpy() == value2
        param1.set_value(value3)
        assert param1.numpy() == value3
        with param1.set_value(value4):
            assert param1.numpy() == value4
        assert param1.numpy() == value3
    assert param1.numpy() == value1


def test_fixed_param():
    obs = zfit.Space("obs1", (-1, 2))
    sigma = zfit.param.ConstantParameter('const1', 5)
    gauss = zfit.pdf.Gauss(1., sigma, obs=obs)
    mu = gauss.params['mu']
    assert isinstance(mu, zfit.param.ConstantParameter)
    assert isinstance(sigma, zfit.param.ConstantParameter)
    assert not sigma.floating
    assert not sigma.independent
    assert sigma.get_dependents() == set()


def test_convert_to_parameter():
    pass  # TODO(Mayou36): add tests

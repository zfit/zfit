#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import tensorflow as tf
from ordered_set import OrderedSet

import zfit
import zfit.z.numpy as znp
from zfit import Parameter, z
from zfit.exception import LogicalUndefinedOperationError
from zfit.param import ComplexParameter, ComposedParameter


def test_complex_param():


    real_part = 1.3
    imag_part = 0.3
    complex_value = real_part + 1j * imag_part

    param1 = ComplexParameter(
        "param1_compl",
        lambda params: znp.array(params[0], dtype=np.complex128) + 1j * imag_part,
        params=[real_part, imag_part],
    )
    some_value = 3.0 * param1**2 - 1.2j
    true_value = 3.0 * complex_value**2 - 1.2j
    assert pytest.approx(some_value, rel=1e-7) == true_value
    assert not param1.get_params()

    # Cartesian complex
    real_part_param = Parameter("real_part_param", real_part)
    imag_part_param = Parameter("imag_part_param", imag_part)
    param2 = ComplexParameter.from_cartesian(
        "param2_compl", real_part_param, imag_part_param
    )
    part1, part2 = param2.get_params()
    part1_val, part2_val = [part1.value(), part2.value()]
    if pytest.approx(real_part) == part1_val:
        assert pytest.approx(imag_part) == part2_val
    elif pytest.approx(real_part) == part2_val:
        assert pytest.approx(imag_part) == part1_val
    else:
        assert False, "one of the if or elif should be the case"

    # Polar complex
    mod_val = 2
    arg_val = np.pi / 4.0
    mod_part_param = Parameter("mod_part_param", mod_val)
    arg_part_param = Parameter("arg_part_param", arg_val)
    param3 = ComplexParameter.from_polar("param3_compl", mod_part_param, arg_part_param)
    part1, part2 = param3.get_params()
    part1_val, part2_val = [part1.value(), part2.value()]
    if pytest.approx(mod_val) == part1_val:
        assert pytest.approx(arg_val) == part2_val
    elif pytest.approx(arg_val) == part1_val:
        assert pytest.approx(mod_val) == part2_val
    else:
        assert False, "one of the if or elif should be the case"

    param4_name = "param4"
    param4 = ComplexParameter.from_polar(param4_name, 4.0, 2.0, floating=True)
    deps_param4 = param4.get_params()
    assert len(deps_param4) == 2
    for dep in deps_param4:
        assert dep.floating

    # Test complex conjugate
    param5 = param2.conj

    # Test properties (1e-8 is too precise)
    assert pytest.approx(real_part, rel=1e-6) == param1.real
    assert pytest.approx(imag_part, rel=1e-6) == param1.imag
    assert pytest.approx(real_part, rel=1e-6) == param2.real
    assert pytest.approx(imag_part, rel=1e-6) == param2.imag
    assert pytest.approx(np.abs(complex_value)) == param2.mod
    assert pytest.approx(np.angle(complex_value)) == param2.arg
    assert pytest.approx(mod_val, rel=1e-6) == param3.mod
    assert pytest.approx(arg_val, rel=1e-6) == param3.arg
    assert pytest.approx(mod_val * np.cos(arg_val), rel=1e-6) == param3.real
    assert pytest.approx(mod_val * np.sin(arg_val), rel=1e-6) == param3.imag
    assert pytest.approx(real_part) == param5.real
    assert pytest.approx(-imag_part) == param5.imag


def test_repr():
    val = 1543
    val2 = 1543**2
    param1 = Parameter("param1", val)
    param2 = zfit.ComposedParameter("comp1", lambda x: x**2, params=param1)
    repr_value = repr(param1)
    repr_value2 = repr(param2)
    assert str(val) in repr_value

    @z.function
    def tf_call():
        repr_value = repr(param1)
        repr_value2 = repr(param2)
        assert str(val) not in repr_value
        assert str(val2) not in repr_value2
        assert "graph-node" in repr_value
        assert "graph-node" in repr_value2

    if zfit.run.get_graph_mode():  # only test if running in graph mode
        tf_call()


def test_composed_param():
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = Parameter("param3", 3.0, floating=False)
    param4 = Parameter(
        "param4", 4.0
    )  # noqa Needed to make sure it does not only take all params as deps

    def func(p1, p2, p3):
        return z.math.log(3.0 * p1) * tf.square(p2) - p3

    param_a = ComposedParameter(
        "param_as", func=func, params=(param1, param2, param3)
    )
    param_a2 = ComposedParameter(
        "param_as2",
        func=func,
        params={f"p{i}": p for i, p in enumerate((param1, param2, param3))},
    )
    assert param_a2.params["p1"] == param2
    assert isinstance(param_a.get_params(floating=True), OrderedSet)
    assert set(param_a.get_params(floating=True)) == {param1, param2}
    assert set(param_a.get_params(floating=None)) == {param1, param2, param3}
    a_unchanged = func(param1, param2, param3)
    assert a_unchanged == param_a.value()
    param2.assign(3.5)
    a_changed = func(param1, param2, param3)
    assert a_changed == param_a
    assert a_changed != a_unchanged

    # Test param representation
    str(param_a)
    repr(param_a)

    @z.function
    def print_param(p):
        _ = str(p)
        _ = repr(p)

    print_param(param_a)

    with pytest.raises(LogicalUndefinedOperationError):
        param_a.set_value(value=5.0)
    with pytest.raises(LogicalUndefinedOperationError):
        param_a.assign(value=5.0)
    with pytest.raises(LogicalUndefinedOperationError):
        param_a.randomize()


def test_shape_parameter():
    a = Parameter(name="a", value=1)
    assert a.shape.rank == 0


def test_shape_composed_parameter():
    a = Parameter(name="a", value=1)
    b = Parameter(name="b", value=2)

    def compose():
        return tf.square(a) - b

    c = ComposedParameter(name="c", func=compose, params=[a, b])
    assert c.shape.rank == 0


# TODO: add test
def test_randomize():
    param1 = zfit.Parameter("param1", 1.0, 0, 2)
    for _ in range(100):
        param1.randomize()
        assert 0 < param1 < 2


def test_floating_behavior():
    param1 = zfit.Parameter("param1", 1.0)
    assert param1.floating


def test_param_limits():
    lower, upper = -4.0, 3.0
    param1 = Parameter("param1", 1.0, lower=lower, upper=upper)
    param2 = Parameter("param2", 2.0)

    assert param1.has_limits
    assert not param2.has_limits

    with pytest.raises(ValueError):
        param1.set_value(upper + 0.5)
    param1.assign(upper + 0.5)
    assert upper == param1.value()
    assert param1.at_limit
    with pytest.raises(ValueError):
        param1.set_value(lower - 1.1)
    param1.assign(lower - 1.1)
    assert lower == param1.value()
    assert param1.at_limit
    param1.set_value(upper - 0.1)
    assert not param1.at_limit

    param2.lower = lower
    param2.assign(lower - 1.1)
    assert lower == param2.value()


def test_overloaded_operators():
    param1 = zfit.Parameter("param1", 5)
    param2 = zfit.Parameter("param2", 3)
    param_a = ComposedParameter("param_a", lambda p1: p1 * 4, params=param1)
    param_b = ComposedParameter("param_b", lambda p2: p2, params=param2)
    param_c = param_a * param_b
    assert not isinstance(param_c, zfit.Parameter)
    param_d = ComposedParameter(
        "param_d", lambda pa, pb: pa + pa * pb**2, params=[param_a, param_b]
    )
    assert param_d.value() == (param_a + param_a * param_b**2)


def test_equal_naming():
    unique_name = "fafdsfds"
    param_unique_name = zfit.Parameter(unique_name, 5.0)
    param_unique_name2 = zfit.Parameter(unique_name, 3.0)
    assert True, "This is new and has to work now :)"


def test_set_value():
    value1 = 1.0
    value2 = 2.0
    value3 = 3.0
    value4 = 4.0
    param1 = zfit.Parameter(name="param1", value=value1)
    assert param1.value() == value1
    with param1.set_value(value2):
        assert param1.value() == value2
        param1.set_value(value3)
        assert param1.value() == value3
        with param1.set_value(value4):
            assert param1.value() == value4
        assert param1.value() == value3
    assert param1.value() == value1


def test_fixed_param():
    obs = zfit.Space("obs1", (-1, 2))
    sigma = zfit.param.ConstantParameter("const1", 5)
    gauss = zfit.pdf.Gauss(1.0, sigma, obs=obs)
    mu = gauss.params["mu"]
    assert isinstance(mu, zfit.param.ConstantParameter)
    assert isinstance(sigma, zfit.param.ConstantParameter)
    assert not sigma.floating
    assert not sigma.independent
    assert set(sigma.get_params()) == set()


def test_convert_to_parameters():
    import zfit

    conv_param1 = zfit.param.convert_to_parameters(
        [23, 10.0, 34, 23], prefer_constant=True
    )
    assert len(conv_param1) == 4
    assert not any(p.floating for p in conv_param1)

    conv_param2 = zfit.param.convert_to_parameters(
        [23, 10.0, 34, 12, 23], prefer_constant=False
    )
    assert all(p.floating for p in conv_param2)

    trueval3 = [23, 10.0, 12, 34, 23]
    truelower3 = list(range(5))
    conv_param3 = zfit.param.convert_to_parameters(
        trueval3, lower=truelower3, prefer_constant=False
    )
    np.testing.assert_allclose(znp.asarray(conv_param3), trueval3)
    np.testing.assert_allclose(znp.asarray([p.lower for p in conv_param3]), truelower3)

    truename4 = ["oe", "myname1", "ue", "eu", "eue"]
    stepsize4 = [23, 1.5, 10.0, 34, 23]
    trueupper4 = [213, 14.0, 1110.0, 314, 213]
    values4 = [23, 12, 121, 34, 23]

    conv_param4dict = zfit.param.convert_to_parameters(
        {
            "value": values4,
            "name": truename4,
            "stepsize": stepsize4,
            "upper": trueupper4,
        },
        prefer_constant=False,
    )
    assert [p.name for p in conv_param4dict] == truename4

    np.testing.assert_allclose(znp.asarray([p.upper for p in conv_param4dict]), trueupper4)
    np.testing.assert_allclose(
        znp.asarray([p.stepsize for p in conv_param4dict]), stepsize4
    )

    truename5 = [name + "_five" for name in truename4]
    conv_param4dict = zfit.param.convert_to_parameters(
        values4,
        name=truename5,
        upper=trueupper4,
        prefer_constant=False,
        stepsize=stepsize4,
    )
    assert [p.name for p in conv_param4dict] == truename5

    np.testing.assert_allclose(znp.asarray([p.upper for p in conv_param4dict]), trueupper4)
    np.testing.assert_allclose(
        znp.asarray([p.stepsize for p in conv_param4dict]), stepsize4
    )


def test_convert_to_parameters_equivalence_to_single_multi():
    import zfit

    conv_param1 = zfit.param.convert_to_parameters([23, 10.0, 34, 23])[1]
    assert pytest.approx(znp.asarray(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameters(
        [23, 10.0, 34, 12, 23], prefer_constant=False
    )[-2]
    assert pytest.approx(znp.asarray(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameters(
        [23, 10.0, 12, 34, 23], lower=list(range(5)), prefer_constant=False
    )[2]
    assert pytest.approx(znp.asarray(conv_param3.lower)) == 2
    assert conv_param3.has_limits

    truename4 = ["oe", "myname1", "ue", "eu", "eue"]
    stepsize4 = [23, 1.5, 10.0, 34, 23]
    conv_param4 = zfit.param.convert_to_parameters(
        [23, 12, 121, 34, 23],
        name=truename4,
        upper=[213, 14.0, 1110.0, 314, 213],
        prefer_constant=False,
        stepsize=stepsize4,
    )[1]
    assert conv_param4.floating
    assert conv_param4.name == "myname1"
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(znp.asarray(conv_param4.stepsize)) == 1.5


def test_convert_to_parameters_equivalence_to_single():
    import zfit

    conv_param1 = zfit.param.convert_to_parameters(10.0)[0]
    assert pytest.approx(znp.asarray(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameters(12.0, prefer_constant=False)[0]
    assert pytest.approx(znp.asarray(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameters(
        12.0, lower=5.0, prefer_constant=False
    )[0]
    assert pytest.approx(znp.asarray(conv_param3.lower)) == 5.0
    assert conv_param3.has_limits

    truename4 = "myname1"
    stepsize4 = 1.5
    conv_param4 = zfit.param.convert_to_parameters(
        12.0, name=truename4, upper=14.0, prefer_constant=False, stepsize=stepsize4
    )[0]
    assert conv_param4.floating
    assert conv_param4.name == truename4
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(znp.asarray(conv_param4.stepsize)) == stepsize4


def test_convert_to_parameter():
    import zfit

    conv_param1 = zfit.param.convert_to_parameter(10.0)
    assert pytest.approx(znp.asarray(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameter(12.0, prefer_constant=False)
    assert pytest.approx(znp.asarray(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameter(
        12.0, lower=5.0, prefer_constant=False
    )
    assert pytest.approx(znp.asarray(conv_param3.lower)) == 5.0
    assert conv_param3.has_limits

    with pytest.raises(ValueError):
        _ = zfit.param.convert_to_parameter(5.0, lower=15.0, prefer_constant=False)

    with pytest.raises(ValueError):
        _ = zfit.param.convert_to_parameter(5.0, upper=1.0, prefer_constant=False)

    truename4 = "myname1"
    stepsize4 = 1.5
    conv_param4 = zfit.param.convert_to_parameter(
        12.0, name=truename4, upper=14.0, prefer_constant=False, stepsize=stepsize4
    )
    assert conv_param4.floating
    assert conv_param4.name == truename4
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(znp.asarray(conv_param4.stepsize)) == stepsize4


def test_set_values():
    import zfit

    init_values = [1, 2, 3]
    second_values = [5, 6, 7]
    params = [zfit.Parameter(f"param_{i}", val) for i, val in enumerate(init_values)]

    with zfit.param.set_values(params, second_values):
        for param, val in zip(params, second_values):
            assert param.value() == val

    for param, val in zip(params, init_values):
        assert param.value() == val

    zfit.param.set_values(params, second_values)
    for param, val in zip(params, second_values):
        assert param.value() == val

    zfit.param.set_values(params, init_values)
    for param, val in zip(params, init_values):
        assert param.value() == val


@pytest.mark.parametrize("addmore", [True, False])
def test_set_values_dict(addmore):
    import zfit

    init_values = [1, 2, 3]
    second_values = [5, 6, 7]
    params = [zfit.Parameter(f"param_{i}", val) for i, val in enumerate(init_values)]

    setvalueparam = {p: v for p, v in zip(params, second_values)}
    setvalue_paramname = {p.name: v for p, v in zip(params, second_values)}
    setvaluemixed = setvalueparam.copy()
    setvaluemixed["param_1"] = params[1]

    if addmore:
        param4 = zfit.Parameter("param4", 4)
        setvalue_paramname[param4.name] = 4
        setvalueparam[param4] = 4
        setvaluemixed[param4] = 4

    zfit.param.set_values(params, setvalueparam)
    for param, val in zip(params, second_values):
        assert pytest.approx(param.value()) == val

    zfit.param.set_values(params, init_values)

    zfit.param.set_values(params, setvalue_paramname)
    for param, val in zip(params, second_values):
        assert pytest.approx(param.value()) == val

    zfit.param.set_values(params, init_values)
    zfit.param.set_values(params, setvaluemixed)
    for param, val in zip(params, second_values):
        assert pytest.approx(param.value()) == val

    zfit.param.set_values(params, init_values)

    too_small_values = setvalueparam.copy()
    too_small_values.pop(params[0])
    with pytest.raises(ValueError):
        zfit.param.set_values(params, too_small_values)

    zfit.param.set_values(params, init_values)
    # try with allow_partial
    zfit.param.set_values(params, too_small_values, allow_partial=True)
    assert params[0].value() == init_values[0]
    assert params[1].value() == second_values[1]
    assert params[2].value() == second_values[2]

    zfit.param.set_values(params, init_values)


def test_deletion():
    def func():
        a = zfit.Parameter("param", 42)
        return True

    assert func()
    assert func()  # this must not raise an error


def test_to_numpy():
    import zfit

    param = zfit.Parameter("param", 42)
    assert znp.asarray(param) == 42

    p1 = zfit.param.ConstantParameter("aoeu1", 15)
    assert znp.asarray(p1) == 15

    p2 = zfit.param.ComposedParameter(
        "aoeu2", lambda params: 2 * params["p1"], params={"p1": p1}
    )
    assert znp.asarray(p2) == 30

def test_parameter_label():

    param1 = zfit.Parameter("param1", 1.0)
    assert param1.label == "param1"

    param2 = zfit.Parameter("param2", 1.0, label="param2_label")
    assert param2.label == "param2_label"

    param3 = zfit.ComposedParameter("param3", lambda x: x, params=param1)
    assert param3.label == "param3"

    param4 = zfit.ComposedParameter("param4", lambda x: x, params=param1, label="param4_label")
    assert param4.label == "param4_label"

    # complex params
    real_part = 1.3
    imag_part = 0.3
    paramc1 = zfit.ComplexParameter.from_polar("paramc1", 4.0, 2.0, label="paramc1_label")
    assert paramc1.label == "paramc1_label"

    paramc2 = zfit.ComplexParameter.from_cartesian("paramc2", zfit.Parameter("real_part_param", real_part), zfit.Parameter("imag_part_param", imag_part), label="paramc2_label")
    assert paramc2.label == "paramc2_label"

    # constant params
    param_const = zfit.param.ConstantParameter("param_const", 5.0)
    assert param_const.label == "param_const"

    param_const2 = zfit.param.ConstantParameter("param_const2", 5.0, label="param_const2_label")
    assert param_const2.label == "param_const2_label"

#  Copyright (c) 2023 zfit
import numpy as np
import pytest
import tensorflow as tf
from ordered_set import OrderedSet

import zfit
from zfit import Parameter, z
from zfit.exception import NameAlreadyTakenError, LogicalUndefinedOperationError
from zfit.param import ComplexParameter, ComposedParameter


def test_complex_param():
    real_part = 1.3
    imag_part = 0.3
    complex_value = real_part + 1j * imag_part

    param1 = ComplexParameter("param1_compl", lambda: complex_value, params=None)
    some_value = 3.0 * param1**2 - 1.2j
    true_value = 3.0 * complex_value**2 - 1.2j
    assert true_value == pytest.approx(some_value.numpy(), rel=1e-8)
    assert not param1.get_params()

    # Cartesian complex
    real_part_param = Parameter("real_part_param", real_part)
    imag_part_param = Parameter("imag_part_param", imag_part)
    param2 = ComplexParameter.from_cartesian(
        "param2_compl", real_part_param, imag_part_param
    )
    part1, part2 = param2.get_params()
    part1_val, part2_val = [part1.value().numpy(), part2.value().numpy()]
    if part1_val == pytest.approx(real_part):
        assert part2_val == pytest.approx(imag_part)
    elif part2_val == pytest.approx(real_part):
        assert part1_val == pytest.approx(imag_part)
    else:
        assert False, "one of the if or elif should be the case"

    # Polar complex
    mod_val = 2
    arg_val = np.pi / 4.0
    mod_part_param = Parameter("mod_part_param", mod_val)
    arg_part_param = Parameter("arg_part_param", arg_val)
    param3 = ComplexParameter.from_polar("param3_compl", mod_part_param, arg_part_param)
    part1, part2 = param3.get_cache_deps()
    part1_val, part2_val = [part1.value().numpy(), part2.value().numpy()]
    if part1_val == pytest.approx(mod_val):
        assert part2_val == pytest.approx(arg_val)
    elif part1_val == pytest.approx(arg_val):
        assert part2_val == pytest.approx(mod_val)
    else:
        assert False, "one of the if or elif should be the case"

    param4_name = "param4"
    param4 = ComplexParameter.from_polar(param4_name, 4.0, 2.0, floating=True)
    deps_param4 = param4.get_cache_deps()
    assert len(deps_param4) == 2
    for dep in deps_param4:
        assert dep.floating

    # Test complex conjugate
    param5 = param2.conj

    # Test properties (1e-8 is too precise)
    assert zfit.run(param1.real) == pytest.approx(real_part, rel=1e-6)
    assert zfit.run(param1.imag) == pytest.approx(imag_part, rel=1e-6)
    assert zfit.run(param2.real) == pytest.approx(real_part, rel=1e-6)
    assert zfit.run(param2.imag) == pytest.approx(imag_part, rel=1e-6)
    assert zfit.run(param2.mod) == pytest.approx(np.abs(complex_value))
    assert zfit.run(param2.arg) == pytest.approx(np.angle(complex_value))
    assert zfit.run(param3.mod) == pytest.approx(mod_val, rel=1e-6)
    assert zfit.run(param3.arg) == pytest.approx(arg_val, rel=1e-6)
    assert zfit.run(param3.real) == pytest.approx(mod_val * np.cos(arg_val), rel=1e-6)
    assert zfit.run(param3.imag) == pytest.approx(mod_val * np.sin(arg_val), rel=1e-6)
    assert zfit.run(param5.real) == pytest.approx(real_part)
    assert zfit.run(param5.imag) == pytest.approx(-imag_part)


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

    def value_fn(p1, p2, p3):
        return z.math.log(3.0 * p1) * tf.square(p2) - p3

    param_a = ComposedParameter(
        "param_as", value_fn=value_fn, params=(param1, param2, param3)
    )
    param_a2 = ComposedParameter(
        "param_as2",
        value_fn=value_fn,
        params={f"p{i}": p for i, p in enumerate((param1, param2, param3))},
    )
    assert param_a2.params["p1"] == param2
    assert isinstance(param_a.get_cache_deps(only_floating=True), OrderedSet)
    assert param_a.get_cache_deps(only_floating=True) == {param1, param2}
    assert param_a.get_cache_deps(only_floating=False) == {param1, param2, param3}
    a_unchanged = value_fn(param1, param2, param3).numpy()
    assert a_unchanged == param_a.numpy()
    param2.assign(3.5)
    assert param2.numpy()
    a_changed = value_fn(param1, param2, param3).numpy()
    assert a_changed == param_a.numpy()
    assert a_changed != a_unchanged

    # Test param representation
    str(param_a)
    repr(param_a)

    @z.function
    def print_param(p):
        _ = str(p)
        _ = repr(p)

    print_param(param_a)

    # TODO(params): reactivate to check?
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

    c = ComposedParameter(name="c", value_fn=compose, params=[a, b])
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
    assert upper == param1.value().numpy()
    assert param1.at_limit
    with pytest.raises(ValueError):
        param1.set_value(lower - 1.1)
    param1.assign(lower - 1.1)
    assert lower == param1.value().numpy()
    assert param1.at_limit
    param1.set_value(upper - 0.1)
    assert not param1.at_limit

    param2.lower = lower
    param2.assign(lower - 1.1)
    assert lower == param2.value().numpy()


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
    param_d_val = param_d.numpy()
    assert param_d_val == (param_a + param_a * param_b**2).numpy()


def test_equal_naming():
    unique_name = "fafdsfds"
    param_unique_name = zfit.Parameter(unique_name, 5.0)
    with pytest.raises(NameAlreadyTakenError):
        param_unique_name2 = zfit.Parameter(unique_name, 3.0)


def test_set_value():
    value1 = 1.0
    value2 = 2.0
    value3 = 3.0
    value4 = 4.0
    param1 = zfit.Parameter(name="param1", value=value1)
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
    sigma = zfit.param.ConstantParameter("const1", 5)
    gauss = zfit.pdf.Gauss(1.0, sigma, obs=obs)
    mu = gauss.params["mu"]
    assert isinstance(mu, zfit.param.ConstantParameter)
    assert isinstance(sigma, zfit.param.ConstantParameter)
    assert not sigma.floating
    assert not sigma.independent
    assert sigma.get_cache_deps() == set()


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
    np.testing.assert_allclose(zfit.run(conv_param3), trueval3)
    np.testing.assert_allclose(zfit.run([p.lower for p in conv_param3]), truelower3)

    truename4 = ["oe", "myname1", "ue", "eu", "eue"]
    stepsize4 = [23, 1.5, 10.0, 34, 23]
    trueupper4 = [213, 14.0, 1110.0, 314, 213]
    values4 = [23, 12, 121, 34, 23]

    conv_param4dict = zfit.param.convert_to_parameters(
        {
            "value": values4,
            "name": truename4,
            "step_size": stepsize4,
            "upper": trueupper4,
        },
        prefer_constant=False,
    )
    assert [p.name for p in conv_param4dict] == truename4

    np.testing.assert_allclose(zfit.run([p.upper for p in conv_param4dict]), trueupper4)
    np.testing.assert_allclose(
        zfit.run([p.step_size for p in conv_param4dict]), stepsize4
    )

    truename5 = [name + "_five" for name in truename4]
    conv_param4dict = zfit.param.convert_to_parameters(
        values4,
        name=truename5,
        upper=trueupper4,
        prefer_constant=False,
        step_size=stepsize4,
    )
    assert [p.name for p in conv_param4dict] == truename5

    np.testing.assert_allclose(zfit.run([p.upper for p in conv_param4dict]), trueupper4)
    np.testing.assert_allclose(
        zfit.run([p.step_size for p in conv_param4dict]), stepsize4
    )


def test_convert_to_parameters_equivalence_to_single_multi():
    import zfit

    conv_param1 = zfit.param.convert_to_parameters([23, 10.0, 34, 23])[1]
    assert pytest.approx(zfit.run(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameters(
        [23, 10.0, 34, 12, 23], prefer_constant=False
    )[-2]
    assert pytest.approx(zfit.run(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameters(
        [23, 10.0, 12, 34, 23], lower=list(range(5)), prefer_constant=False
    )[2]
    assert pytest.approx(zfit.run(conv_param3.lower)) == 2
    assert conv_param3.has_limits

    truename4 = ["oe", "myname1", "ue", "eu", "eue"]
    stepsize4 = [23, 1.5, 10.0, 34, 23]
    conv_param4 = zfit.param.convert_to_parameters(
        [23, 12, 121, 34, 23],
        name=truename4,
        upper=[213, 14.0, 1110.0, 314, 213],
        prefer_constant=False,
        step_size=stepsize4,
    )[1]
    assert conv_param4.floating
    assert conv_param4.name == "myname1"
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(zfit.run(conv_param4.step_size)) == 1.5


def test_convert_to_parameters_equivalence_to_single():
    import zfit

    conv_param1 = zfit.param.convert_to_parameters(10.0)[0]
    assert pytest.approx(zfit.run(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameters(12.0, prefer_constant=False)[0]
    assert pytest.approx(zfit.run(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameters(
        12.0, lower=5.0, prefer_constant=False
    )[0]
    assert pytest.approx(zfit.run(conv_param3.lower)) == 5.0
    assert conv_param3.has_limits

    truename4 = "myname1"
    stepsize4 = 1.5
    conv_param4 = zfit.param.convert_to_parameters(
        12.0, name=truename4, upper=14.0, prefer_constant=False, step_size=stepsize4
    )[0]
    assert conv_param4.floating
    assert conv_param4.name == truename4
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(zfit.run(conv_param4.step_size)) == stepsize4


def test_convert_to_parameter():
    import zfit

    conv_param1 = zfit.param.convert_to_parameter(10.0)
    assert pytest.approx(zfit.run(conv_param1.value())) == 10
    assert not conv_param1.floating

    conv_param2 = zfit.param.convert_to_parameter(12.0, prefer_constant=False)
    assert pytest.approx(zfit.run(conv_param2.value())) == 12.0
    assert conv_param2.floating
    assert not conv_param2.has_limits

    conv_param3 = zfit.param.convert_to_parameter(
        12.0, lower=5.0, prefer_constant=False
    )
    assert pytest.approx(zfit.run(conv_param3.lower)) == 5.0
    assert conv_param3.has_limits

    with pytest.raises(ValueError):
        _ = zfit.param.convert_to_parameter(5.0, lower=15.0, prefer_constant=False)

    with pytest.raises(ValueError):
        _ = zfit.param.convert_to_parameter(5.0, upper=1.0, prefer_constant=False)

    truename4 = "myname1"
    stepsize4 = 1.5
    conv_param4 = zfit.param.convert_to_parameter(
        12.0, name=truename4, upper=14.0, prefer_constant=False, step_size=stepsize4
    )
    assert conv_param4.floating
    assert conv_param4.name == truename4
    assert conv_param4.has_limits
    assert conv_param4.floating
    assert pytest.approx(zfit.run(conv_param4.step_size)) == stepsize4


def test_set_values():
    import zfit

    init_values = [1, 2, 3]
    second_values = [5, 6, 7]
    params = [zfit.Parameter(f"param_{i}", val) for i, val in enumerate(init_values)]

    with zfit.param.set_values(params, second_values):
        for param, val in zip(params, second_values):
            assert param.value().numpy() == val

    for param, val in zip(params, init_values):
        assert param.value().numpy() == val

    zfit.param.set_values(params, second_values)
    for param, val in zip(params, second_values):
        assert param.value().numpy() == val

    zfit.param.set_values(params, init_values)
    for param, val in zip(params, init_values):
        assert param.value().numpy() == val


def test_deletion():
    import gc

    def func():
        a = zfit.Parameter("param", 42)
        return True

    assert func()

    gc.collect()
    assert func()  # this must not raise an error


def test_to_numpy():
    import zfit

    param = zfit.Parameter("param", 42)
    assert zfit.run(param) == 42

    p1 = zfit.param.ConstantParameter("aoeu1", 15)
    assert zfit.run(p1) == 15

    p2 = zfit.param.ComposedParameter(
        "aoeu2", lambda params: 2 * params["p1"], {"p1": p1}
    )
    assert zfit.run(p2) == 30

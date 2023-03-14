#  Copyright (c) 2023 zfit
from contextlib import suppress

import numpy as np
import pytest
import tensorflow as tf

import zfit
import zfit.core.integration as zintegrate
import zfit.z.numpy as znp
from zfit import z
from zfit.core.basepdf import BasePDF
from zfit.core.parameter import Parameter
from zfit.core.space import Space
from zfit.models.dist_tfp import Gauss

limits1_5deps = [((1.0, -1.0, 2.0, 4.0, 3.0),), ((5.0, 4.0, 5.0, 8.0, 9.0),)]
limits_simple_5deps = [((1.0, -1.0, -5.0, 3.4, 2.1),), ((5.0, 5.4, -1.1, 7.6, 3.5),)]

obs1 = "obs1"


def func1_5deps(x):
    a, b, c, d, e = z.unstack_x(x)
    return a + b * c**2 + d**2 * e**3


def func1_5deps_fully_integrated(limits):
    lower, upper = limits
    lower, upper = lower[0], upper[0]
    a_lower, b_lower, c_lower, d_lower, e_lower = lower
    a_upper, b_upper, c_upper, d_upper, e_upper = upper

    val = (
        -(e_lower**4)
        * (
            a_lower * b_lower * c_lower * d_lower**3 / 12
            - a_lower * b_lower * c_lower * d_upper**3 / 12
            - a_lower * b_lower * c_upper * d_lower**3 / 12
            + a_lower * b_lower * c_upper * d_upper**3 / 12
            - a_lower * b_upper * c_lower * d_lower**3 / 12
            + a_lower * b_upper * c_lower * d_upper**3 / 12
            + a_lower * b_upper * c_upper * d_lower**3 / 12
            - a_lower * b_upper * c_upper * d_upper**3 / 12
            - a_upper * b_lower * c_lower * d_lower**3 / 12
            + a_upper * b_lower * c_lower * d_upper**3 / 12
            + a_upper * b_lower * c_upper * d_lower**3 / 12
            - a_upper * b_lower * c_upper * d_upper**3 / 12
            + a_upper * b_upper * c_lower * d_lower**3 / 12
            - a_upper * b_upper * c_lower * d_upper**3 / 12
            - a_upper * b_upper * c_upper * d_lower**3 / 12
            + a_upper * b_upper * c_upper * d_upper**3 / 12
        )
        - e_lower
        * (
            a_lower**2 * b_lower * c_lower * d_lower / 2
            - a_lower**2 * b_lower * c_lower * d_upper / 2
            - a_lower**2 * b_lower * c_upper * d_lower / 2
            + a_lower**2 * b_lower * c_upper * d_upper / 2
            - a_lower**2 * b_upper * c_lower * d_lower / 2
            + a_lower**2 * b_upper * c_lower * d_upper / 2
            + a_lower**2 * b_upper * c_upper * d_lower / 2
            - a_lower**2 * b_upper * c_upper * d_upper / 2
            + a_lower * b_lower**2 * c_lower**3 * d_lower / 6
            - a_lower * b_lower**2 * c_lower**3 * d_upper / 6
            - a_lower * b_lower**2 * c_upper**3 * d_lower / 6
            + a_lower * b_lower**2 * c_upper**3 * d_upper / 6
            - a_lower * b_upper**2 * c_lower**3 * d_lower / 6
            + a_lower * b_upper**2 * c_lower**3 * d_upper / 6
            + a_lower * b_upper**2 * c_upper**3 * d_lower / 6
            - a_lower * b_upper**2 * c_upper**3 * d_upper / 6
            - a_upper**2 * b_lower * c_lower * d_lower / 2
            + a_upper**2 * b_lower * c_lower * d_upper / 2
            + a_upper**2 * b_lower * c_upper * d_lower / 2
            - a_upper**2 * b_lower * c_upper * d_upper / 2
            + a_upper**2 * b_upper * c_lower * d_lower / 2
            - a_upper**2 * b_upper * c_lower * d_upper / 2
            - a_upper**2 * b_upper * c_upper * d_lower / 2
            + a_upper**2 * b_upper * c_upper * d_upper / 2
            - a_upper * b_lower**2 * c_lower**3 * d_lower / 6
            + a_upper * b_lower**2 * c_lower**3 * d_upper / 6
            + a_upper * b_lower**2 * c_upper**3 * d_lower / 6
            - a_upper * b_lower**2 * c_upper**3 * d_upper / 6
            + a_upper * b_upper**2 * c_lower**3 * d_lower / 6
            - a_upper * b_upper**2 * c_lower**3 * d_upper / 6
            - a_upper * b_upper**2 * c_upper**3 * d_lower / 6
            + a_upper * b_upper**2 * c_upper**3 * d_upper / 6
        )
        + e_upper**4
        * (
            a_lower * b_lower * c_lower * d_lower**3 / 12
            - a_lower * b_lower * c_lower * d_upper**3 / 12
            - a_lower * b_lower * c_upper * d_lower**3 / 12
            + a_lower * b_lower * c_upper * d_upper**3 / 12
            - a_lower * b_upper * c_lower * d_lower**3 / 12
            + a_lower * b_upper * c_lower * d_upper**3 / 12
            + a_lower * b_upper * c_upper * d_lower**3 / 12
            - a_lower * b_upper * c_upper * d_upper**3 / 12
            - a_upper * b_lower * c_lower * d_lower**3 / 12
            + a_upper * b_lower * c_lower * d_upper**3 / 12
            + a_upper * b_lower * c_upper * d_lower**3 / 12
            - a_upper * b_lower * c_upper * d_upper**3 / 12
            + a_upper * b_upper * c_lower * d_lower**3 / 12
            - a_upper * b_upper * c_lower * d_upper**3 / 12
            - a_upper * b_upper * c_upper * d_lower**3 / 12
            + a_upper * b_upper * c_upper * d_upper**3 / 12
        )
        + e_upper
        * (
            a_lower**2 * b_lower * c_lower * d_lower / 2
            - a_lower**2 * b_lower * c_lower * d_upper / 2
            - a_lower**2 * b_lower * c_upper * d_lower / 2
            + a_lower**2 * b_lower * c_upper * d_upper / 2
            - a_lower**2 * b_upper * c_lower * d_lower / 2
            + a_lower**2 * b_upper * c_lower * d_upper / 2
            + a_lower**2 * b_upper * c_upper * d_lower / 2
            - a_lower**2 * b_upper * c_upper * d_upper / 2
            + a_lower * b_lower**2 * c_lower**3 * d_lower / 6
            - a_lower * b_lower**2 * c_lower**3 * d_upper / 6
            - a_lower * b_lower**2 * c_upper**3 * d_lower / 6
            + a_lower * b_lower**2 * c_upper**3 * d_upper / 6
            - a_lower * b_upper**2 * c_lower**3 * d_lower / 6
            + a_lower * b_upper**2 * c_lower**3 * d_upper / 6
            + a_lower * b_upper**2 * c_upper**3 * d_lower / 6
            - a_lower * b_upper**2 * c_upper**3 * d_upper / 6
            - a_upper**2 * b_lower * c_lower * d_lower / 2
            + a_upper**2 * b_lower * c_lower * d_upper / 2
            + a_upper**2 * b_lower * c_upper * d_lower / 2
            - a_upper**2 * b_lower * c_upper * d_upper / 2
            + a_upper**2 * b_upper * c_lower * d_lower / 2
            - a_upper**2 * b_upper * c_lower * d_upper / 2
            - a_upper**2 * b_upper * c_upper * d_lower / 2
            + a_upper**2 * b_upper * c_upper * d_upper / 2
            - a_upper * b_lower**2 * c_lower**3 * d_lower / 6
            + a_upper * b_lower**2 * c_lower**3 * d_upper / 6
            + a_upper * b_lower**2 * c_upper**3 * d_lower / 6
            - a_upper * b_lower**2 * c_upper**3 * d_upper / 6
            + a_upper * b_upper**2 * c_lower**3 * d_lower / 6
            - a_upper * b_upper**2 * c_lower**3 * d_upper / 6
            - a_upper * b_upper**2 * c_upper**3 * d_lower / 6
            + a_upper * b_upper**2 * c_upper**3 * d_upper / 6
        )
    )
    return val


limits2 = (-1.0, 2.0)
limits2_split = [(-1.0, 1.5), (1.5, 2.0)]


def func2_1deps(x):
    a = x
    return a**2


def func2_1deps_fully_integrated(limits):
    lower, upper = limits
    with suppress(TypeError):
        lower, upper = lower[0], upper[0]

    def func_int(x):
        return (1 / 3) * x**3

    return func_int(upper) - func_int(lower)


limits3 = [((-1.0, -4.3),), ((2.3, -1.2),)]


def func3_2deps(x):
    a, b = z.unstack_x(x)
    return a**2 + b**2


def func3_2deps_fully_integrated(limits, params=None, model=None):
    lower, upper = limits.rect_limits
    with suppress(TypeError):
        lower, upper = lower[0], upper[0]

    lower_a, lower_b = lower
    upper_a, upper_b = upper
    integral = (lower_a**3 - upper_a**3) * (lower_b - upper_b)
    integral += (lower_a - upper_a) * (lower_b**3 - upper_b**3)
    integral /= 3
    return z.convert_to_tensor(integral)


limits4_2dim = [((-4.0, 1.0),), ((-1.0, 4.5),)]
limits4_1dim = (-2.0, 3.0)

func4_values = np.array([-12.0, -4.5, 1.9, 4.1])
func4_2values = np.array([[-12.0, -4.5, 1.9, 4.1], [-11.0, 3.2, 7.4, -0.3]])


def func4_3deps(x):
    if isinstance(x, np.ndarray):
        a, b, c = x
    else:
        a, b, c = z.unstack_x(x)

    return a**2 + b**3 + 0.5 * c


def func4_3deps_0and2_integrated(x, limits):
    b = x
    lower, upper = limits
    a_lower, c_lower = lower[0]
    a_upper, c_upper = upper[0]
    integral = (
        -(c_lower**2) * (-0.25 * a_lower + 0.25 * a_upper)
        - c_lower
        * (
            -0.333333333333333 * a_lower**3
            - 1.0 * a_lower * b**3
            + 0.333333333333333 * a_upper**3
            + 1.0 * a_upper * b**3
        )
        + c_upper**2 * (-0.25 * a_lower + 0.25 * a_upper)
        + c_upper
        * (
            -0.333333333333333 * a_lower**3
            - 1.0 * a_lower * b**3
            + 0.333333333333333 * a_upper**3
            + 1.0 * a_upper * b**3
        )
    )

    return integral


def func4_3deps_1_integrated(x, limits):
    a, c = x
    b_lower, b_upper = limits
    with suppress(TypeError):
        b_lower, b_upper = b_lower[0], b_upper[0]

    integral = (
        -0.25 * b_lower**4
        - b_lower * (1.0 * a**2 + 0.5 * c)
        + 0.25 * b_upper**4
        + b_upper * (1.0 * a**2 + 0.5 * c)
    )
    return integral


@pytest.mark.parametrize("chunksize", [10000000, 1000])
@pytest.mark.parametrize("limits", [limits2, limits2_split])
def test_mc_integration(chunksize, limits):
    # simpel example
    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize
    num_integral = zintegrate.mc_integrate(
        func=func1_5deps,
        limits=Space(limits=limits_simple_5deps, axes=tuple(range(5))),
        n_axes=5,
    )
    if isinstance(limits, list):
        spaces = [Space(limits=limit, axes=tuple(range(1))) for limit in limits]
        space2 = spaces[0] + spaces[1]
    else:
        space2 = Space(limits=limits2, axes=tuple(range(1)))
    num_integral2 = zintegrate.mc_integrate(func=func2_1deps, limits=space2, n_axes=1)
    num_integral3 = zintegrate.mc_integrate(
        func=func3_2deps, limits=Space(limits=limits3, axes=(0, 1)), n_axes=2
    )

    integral = num_integral.numpy()
    integral2 = num_integral2.numpy()
    integral3 = num_integral3.numpy()

    assert integral.shape == (1,)
    assert integral2.shape == (1,)
    assert integral3.shape == (1,)
    assert func1_5deps_fully_integrated(limits_simple_5deps) == pytest.approx(
        integral, rel=0.1
    )
    assert func2_1deps_fully_integrated(limits2) == pytest.approx(integral2, rel=0.03)
    assert func3_2deps_fully_integrated(
        Space(limits=limits3, axes=(0, 1))
    ).numpy() == pytest.approx(integral3, rel=0.03)


@pytest.mark.flaky(2)
def test_mc_partial_integration():
    values = z.convert_to_tensor(func4_values)
    data1 = zfit.Data.from_tensor(obs="obs2", tensor=tf.expand_dims(values, axis=-1))
    limits1 = Space(limits=limits4_2dim, obs=["obs1", "obs3"], axes=(0, 2))
    num_integral = zintegrate.mc_integrate(func=func4_3deps, limits=limits1, x=data1)

    vals_tensor = z.convert_to_tensor(func4_2values)

    vals_reshaped = tf.transpose(a=vals_tensor)
    data2 = zfit.Data.from_tensor(obs=["obs1", "obs3"], tensor=vals_reshaped)

    limits2 = Space(limits=limits4_1dim, obs=["obs2"], axes=1)
    num_integral2 = zintegrate.mc_integrate(
        func=func4_3deps, limits=limits2, x=data2, draws_per_dim=1000
    )

    integral = num_integral.numpy()
    integral2 = num_integral2.numpy()
    assert len(integral) == len(func4_values)
    assert len(integral2) == len(func4_2values[0])
    assert func4_3deps_0and2_integrated(
        x=func4_values, limits=limits4_2dim
    ) == pytest.approx(integral, rel=0.05)

    assert func4_3deps_1_integrated(
        x=func4_2values, limits=limits4_1dim
    ) == pytest.approx(integral2, rel=0.05)


def test_analytic_integral():
    class DistFunc3(BasePDF):
        def _unnormalized_pdf(self, x):
            return func3_2deps(x)

    class CustomGaussOLD(BasePDF):
        def __init__(self, mu, sigma, obs, name="Gauss"):
            super().__init__(name=name, obs=obs, params=dict(mu=mu, sigma=sigma))

        def _unnormalized_pdf(self, x):
            x = x.unstack_x()
            mu = self.params["mu"]
            sigma = self.params["sigma"]
            gauss = znp.exp(-0.5 * tf.square((x - mu) / sigma))

            return gauss

    def _gauss_integral_from_inf_to_inf(limits, params, model):
        return tf.sqrt(2 * znp.pi) * params["sigma"]

    CustomGaussOLD.register_analytic_integral(
        func=_gauss_integral_from_inf_to_inf,
        limits=Space(limits=(-np.inf, np.inf), axes=(0,)),
    )

    mu_true = 1.4
    sigma_true = 1.8
    mu = Parameter("mu_1414", mu_true, mu_true - 2.0, mu_true + 7.0)
    sigma = Parameter("sigma_1414", sigma_true, sigma_true - 10.0, sigma_true + 5.0)
    gauss_params1 = CustomGaussOLD(mu=mu, sigma=sigma, obs=obs1, name="gauss_params1")
    normal_params1 = Gauss(mu=mu, sigma=sigma, obs=obs1, name="gauss_params1")
    gauss_integral_infs = gauss_params1.integrate(
        limits=(-8 * sigma_true, 8 * sigma_true), norm=False
    )
    normal_integral_infs = normal_params1.integrate(
        limits=(-8 * sigma_true, 8 * sigma_true),
        norm=False,
    )

    DistFunc3.register_analytic_integral(
        func=func3_2deps_fully_integrated, limits=Space(limits=limits3, axes=(0, 1))
    )

    dist_func3 = DistFunc3(obs=["obs1", "obs2"])
    normal_integral_infs = normal_integral_infs
    func3_integrated = dist_func3.integrate(
        limits=Space(limits=limits3, axes=(0, 1)),
        norm=False,
    ).numpy()
    assert func3_integrated == pytest.approx(
        func3_2deps_fully_integrated(limits=Space(limits=limits3, axes=(0, 1))).numpy()
    )
    assert gauss_integral_infs.numpy() == pytest.approx(
        np.sqrt(np.pi * 2.0) * sigma_true, rel=0.0001
    )
    assert normal_integral_infs.numpy() == pytest.approx(1, rel=0.0001)


def test_analytic_integral_selection():
    class DistFuncInts(BasePDF):
        def _unnormalized_pdf(self, x):
            return x**2

    int1 = (
        lambda x: 1
    )  # on purpose wrong signature but irrelevant (not called, only test bookkeeping)
    int2 = lambda x: 2
    int22 = lambda x: 22
    int3 = lambda x: 3
    int4 = lambda x: 4
    int5 = lambda x: 5
    limits1 = (-1, 5)
    dims1 = (1,)
    limits1 = Space(axes=dims1, limits=limits1)
    limits2 = (Space.ANY_LOWER, 5)
    dims2 = (1,)
    limits2 = Space(axes=dims2, limits=limits2)
    limits3 = ((Space.ANY_LOWER, 1),), ((Space.ANY_UPPER, 5),)
    dims3 = (0, 1)
    limits3 = Space(axes=dims3, limits=limits3)
    limits4 = (((Space.ANY_LOWER, 0, Space.ANY_LOWER),), ((Space.ANY_UPPER, 5, 42),))
    dims4 = (0, 1, 2)
    limits4 = Space(axes=dims4, limits=limits4)
    limits5 = (((Space.ANY_LOWER, 1),), ((10, Space.ANY_UPPER),))
    dims5 = (1, 2)
    limits5 = Space(axes=dims5, limits=limits5)
    DistFuncInts.register_analytic_integral(int1, limits=limits1)
    DistFuncInts.register_analytic_integral(int2, limits=limits2)
    DistFuncInts.register_analytic_integral(int22, limits=limits2, priority=60)
    DistFuncInts.register_analytic_integral(int3, limits=limits3)
    DistFuncInts.register_analytic_integral(int4, limits=limits4)
    DistFuncInts.register_analytic_integral(int5, limits=limits5)
    dims = DistFuncInts._analytic_integral.get_max_axes(
        limits=Space(limits=(((-5, 1),), ((1, 5),)), axes=dims3)
    )
    assert dims3 == dims

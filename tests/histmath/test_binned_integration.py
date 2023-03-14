#  Copyright (c) 2023 zfit
import numpy as np
import pytest

import zfit
from zfit.core.binnedpdf import binned_rect_integration


def test_calculate_scaled_edges(edges_bins1):
    from zfit.core.binnedpdf import cut_edges_and_bins

    edges, true_scaled_edges, limits, limits_true, value_scaling, values = edges_bins1
    scaled_edges, bins, _ = cut_edges_and_bins(edges, limits)
    np.testing.assert_allclose(true_scaled_edges[0], scaled_edges[0])
    np.testing.assert_allclose(true_scaled_edges[1], scaled_edges[1])
    np.testing.assert_allclose(true_scaled_edges[2], scaled_edges[2])


def test_binned_rect_integration(edges_bins1):
    edges, true_scaled_edges, limits, limits_true, value_scaling, values = edges_bins1

    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = limits_true.area() * value_scaling
    assert pytest.approx(float(true_integral)) == float(integral)

    # integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    # true_integral = value_scaling * np.prod([e.shape.num_elements()
    #                                                               for e in true_scaled_edges])
    # assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_simple():
    import zfit.z.numpy as znp

    scaling = 0.2
    values = znp.ones((2, 2)) * scaling
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (0, 5))
    lim2 = zfit.Space("b", (1, 9))
    limits = lim1 * lim2
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = 5 * 8 * scaling  # area lim1, area lim2, scaling
    assert pytest.approx(float(true_integral)) == float(integral)

    integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    true_integral = 4 * scaling  # 4 elements

    assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_simple_too_large():
    import zfit.z.numpy as znp

    scaling = 0.2
    values = znp.ones((2, 2)) * scaling
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (-1, 10))
    lim2 = zfit.Space("b", (0.5, 19))
    limits = lim1 * lim2
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = 5 * 8 * scaling  # area lim1, area lim2, scaling
    assert pytest.approx(float(true_integral)) == float(integral)

    integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    true_integral = 4 * scaling  # 4 elements

    assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_simple_scaled():
    import zfit.z.numpy as znp

    scaling = 0.2
    values = znp.ones((2, 2)) * scaling
    reducefac = 0.8  # reduce the limits by a factor of 0.8
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (0.5, 4.5))
    lim2 = zfit.Space("b", (1.8, 8.2))
    limits = lim1 * lim2
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = 5 * 8 * scaling * reducefac**2  # area lim1, area lim2, scaling
    assert pytest.approx(float(true_integral)) == float(integral)

    integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    true_integral = (2 * reducefac) ** 2 * scaling  # 4 elements

    assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_simple_scaled_asym():
    import zfit.z.numpy as znp

    scaling = 0.2
    values = znp.ones((2, 2)) * scaling
    reducefac = 0.85  # reduce the limits by a factor of 0.8
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (0.25, 4.5))
    lim2 = zfit.Space("b", (1.4, 8.2))
    limits = lim1 * lim2
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = 5 * 8 * scaling * reducefac**2  # area lim1, area lim2, scaling
    assert pytest.approx(float(true_integral)) == float(integral)

    integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    true_integral = (2 * reducefac) ** 2 * scaling  # 4 elements

    assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_scaled_asym():
    import zfit.z.numpy as znp

    values = znp.array([[1.0, 3.0], [2.0, 4.0]])
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (1.2, 4.7))
    lim2 = zfit.Space("b", (2.1, 8.2))
    limits = lim1 * lim2
    binw11 = 1.3
    binw21 = 2.9
    binw12 = 2.2
    binw22 = 3.2

    # using densities
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = (
        1 * binw11 * binw21
        + 2 * binw12 * binw21
        + 3 * binw11 * binw22
        + 4 * binw12 * binw22
    )
    assert pytest.approx(float(true_integral)) == float(integral)

    # using counts
    integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    true_integral = (
        1 * binw11 * binw21
        + 2 * binw12 * binw21
        + 3 * binw11 * binw22
        + 4 * binw12 * binw22
    )
    true_integral /= 10  # each bin has an area of 10
    assert pytest.approx(float(true_integral)) == float(integral)


def test_binned_partial_scaled_asym_axis0():
    import zfit.z.numpy as znp

    values = znp.array([[1.0, 3.0], [2.0, 4.0]])
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (1.2, 4.7))
    integral = binned_rect_integration(density=values, edges=edges, limits=lim1, axis=0)
    binw11 = 1.3
    binw21 = 4
    binw12 = 2.2
    binw22 = 4
    true_integral = (
        1 * binw11 * binw21 + 2 * binw12 * binw21,
        3 * binw11 * binw22 + 4 * binw12 * binw22,
    )
    assert pytest.approx(true_integral) == zfit.run(integral)


def test_binned_partial_scaled_asym_axis1():
    import zfit.z.numpy as znp

    values = znp.array([[1.0, 3.0], [2.0, 4.0]])
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim2 = zfit.Space("b", (2.1, 8.2))

    integral = binned_rect_integration(density=values, edges=edges, limits=lim2, axis=1)
    binw11 = 2.5
    binw21 = 2.9
    binw12 = 2.5
    binw22 = 3.2
    true_integral = (
        1 * binw11 * binw21 + 3 * binw11 * binw22,
        2 * binw12 * binw21 + 4 * binw12 * binw22,
    )
    assert pytest.approx(true_integral) == zfit.run(integral)


def test_binned_scaled_asym_one():
    import zfit.z.numpy as znp

    values = znp.array([[1.0, 3.0], [2.0, 4.0]])
    edges = [[0.0, 2.5, 5.0], [1.0, 5.0, 9.0]]
    lim1 = zfit.Space("a", (0, 4.7))
    lim2 = zfit.Space("b", (1, 9))
    limits = lim1 * lim2
    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    binw11 = 2.5
    binw21 = 4
    binw12 = 2.2
    binw22 = 4
    true_integral = (
        1 * binw11 * binw21
        + 2 * binw12 * binw21
        + 3 * binw11 * binw22
        + 4 * binw12 * binw22
    )
    assert pytest.approx(float(true_integral)) == float(integral)

    # integral = binned_rect_integration(counts=values, edges=edges, limits=limits)
    # true_integral = (2 * reducefac) ** 2 * scaling  # 4 e


def test_partial_binned_rect_integration(edges_bins1):
    edges, true_scaled_edges, limits, limits_true, value_scaling, values = edges_bins1
    limits_part = limits.with_obs(["a", "c"])
    limits_nonint = limits.with_obs("b")
    limits_nonint = limits_nonint.with_limits([1.9, 4.2])
    limits_full = (limits_part * limits_nonint).with_obs(["a", "b", "c"])
    integral = binned_rect_integration(
        density=values, edges=edges, limits=limits_part, axis=[0, 2]
    )
    assert integral.shape[0] == edges[1].shape[1] - 1
    full_integral = binned_rect_integration(
        density=values, edges=edges, limits=limits_full
    )
    assert pytest.approx(np.sum(integral)) == float(full_integral)


@pytest.fixture()
def edges_bins1():
    import zfit
    import zfit.z.numpy as znp

    lower3 = 1.5
    upper3 = 2.5
    num1 = 6
    num2 = 10
    num3 = 12
    value_scaling = 5.2
    values = znp.ones(shape=(num1 - 1, num2 - 1, num3 - 1)) * value_scaling
    edges3 = znp.linspace(lower3, upper3, num=num3)[None, None, ...]
    edges = [
        znp.linspace(0, 5, num=num1)[..., None, None],
        znp.linspace(2, 4, num=num2)[None, ..., None],
        edges3,
    ]
    lower1 = 1.5
    upper1 = 4.1
    lower2 = 2.5
    upper2 = 4.2
    true_scaled_edges = [
        znp.array([lower1, 2, 3, 4, upper1])[..., None, None],
        znp.array(
            [
                lower2,
                2.66666667,
                2.88888889,
                3.11111111,
                3.33333333,
                3.55555556,
                3.77777778,
                4.0,
            ]
        )[None, ..., None],
        edges3,
    ]
    limits = (
        zfit.Space(obs=["a"], limits=[lower1, upper1])
        * zfit.Space(["b"], [lower2, upper2])
        * zfit.Space(["c"], [lower3, upper3])
    )

    limits_true = (
        zfit.Space(obs=["a"], limits=[lower1, upper1])
        * zfit.Space(["b"], [lower2, 4])
        * zfit.Space(["c"], [lower3, upper3])
    )

    return edges, true_scaled_edges, limits, limits_true, value_scaling, values

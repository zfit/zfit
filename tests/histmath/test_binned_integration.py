#  Copyright (c) 2021 zfit
import hist
import numpy as np
import pytest

from zfit.core.binnedpdf import binned_rect_integration


def test_calculate_scaled_edges(edges_bins1):
    from zfit.core.binnedpdf import cut_edges_and_bins

    edges, true_scaled_edges, limits, limits_true, value_scaling, values = edges_bins1
    scaled_edges, bins = cut_edges_and_bins(edges, limits)
    np.testing.assert_allclose(true_scaled_edges[0], scaled_edges[0])
    np.testing.assert_allclose(true_scaled_edges[1], scaled_edges[1])
    np.testing.assert_allclose(true_scaled_edges[2], scaled_edges[2])


def test_binned_rect_integration(edges_bins1):
    edges, _, limits, limits_true, value_scaling, values = edges_bins1

    integral = binned_rect_integration(density=values, edges=edges, limits=limits)
    true_integral = limits_true.area() * value_scaling
    assert pytest.approx(float(true_integral)) == float(integral)


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
    values = znp.ones(shape=(num1, num2, num3)) * value_scaling
    edges3 = znp.linspace(lower3, upper3, num=num3)[None, None, ...]
    edges = [
        znp.linspace(0, 5, num=num1)[..., None, None],
        znp.linspace(2, 4, num=num2)[None, ..., None],
        edges3
    ]
    lower1 = 1.5
    upper1 = 4.1
    lower2 = 2.5
    upper2 = 4.2
    true_scaled_edges = [

        znp.array([lower1, 2, 3, 4, upper1])[..., None, None],
        znp.array(
            [lower2, 2.66666667, 2.88888889,
             3.11111111, 3.33333333, 3.55555556, 3.77777778, 4.]
        )[None, ..., None],
        edges3

    ]
    limits = (zfit.Space(obs=['a'], limits=[lower1, upper1])
              * zfit.Space(['b'], [lower2, upper2])
              * zfit.Space(['c'], [lower3, upper3]))

    limits_true = (zfit.Space(obs=['a'], limits=[lower1, upper1])
                   * zfit.Space(['b'], [lower2, 4])
                   * zfit.Space(['c'], [lower3, upper3]))

    return edges, true_scaled_edges, limits, limits_true, value_scaling, values
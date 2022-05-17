#  Copyright (c) 2022 zfit
import pytest


@pytest.fixture
def params():
    import zfit

    return [zfit.Parameter(f"param_{i}", i) for i in range(3)]


def test_approx(params):
    import numpy as np

    from zfit.result import Approximations

    grad = np.array(range(3))
    hessian = np.random.normal(size=(3, 3)) + 15
    approx = Approximations(params=params, gradient=grad, hessian=hessian)
    np.testing.assert_allclose(approx.gradient(), grad)
    np.testing.assert_allclose(approx.gradient(params=params), grad)
    np.testing.assert_allclose(approx.gradient(params=params[1]), grad[1])
    np.testing.assert_allclose(approx.gradient(params=params[2:0:-1]), grad[2:0:-1])

    np.testing.assert_allclose(approx.hessian(), hessian)
    assert approx.inv_hessian(invert=False) is None
    np.testing.assert_allclose(approx.inv_hessian(invert=True), np.linalg.inv(hessian))
    # now it should be available and cached
    np.testing.assert_allclose(approx.inv_hessian(invert=False), np.linalg.inv(hessian))

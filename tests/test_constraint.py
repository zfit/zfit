#  Copyright (c) 2019 zfit
import pytest
import numpy as np

import zfit
from zfit import ztf
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.exception import ShapeIncompatibleError
from zfit.core.constraint import BaseConstraint, SimpleConstraint, GaussianConstraint
from zfit.util.container import convert_to_container


def true_nll_gaussian(params, mu, sigma):
    params = convert_to_container(params, container=tuple)
    mu = convert_to_container(mu, container=tuple)
    sigma = convert_to_container(sigma, container=tuple)
    constraint = ztf.constant(0.)
    if not len(params) == len(mu) == len(sigma):
        raise ValueError("params, mu and sigma have to have the same length.")
    for param, mean, sig in zip(params, mu, sigma):
        constraint += ztf.reduce_sum(ztf.square(param - mean) / (2. * ztf.square(sig)))

    return constraint


def test_base_constraint():
    with pytest.raises(TypeError):
        _ = BaseConstraint()


def test_gaussian_constraint_shape_errors():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)

    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint([param1, param2], mu=[4, 2, 3], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint([param1, param2], mu=[4, 2], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint([param1, param2], mu=2, sigma=[1, 4])
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(param1, mu=[4, 2], sigma=[2, 3])


def test_gaussian_constraint():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    mu = [3., 6.1]
    sigma = np.array([[1, 0.3],
                      [0.3, 0.5]])
    constr = GaussianConstraint(params=params, mu=mu, sigma=sigma)
    constr_np = zfit.run(constr.value())
    assert constr_np == pytest.approx(3.989638)

    assert constr.get_dependents() == set(params)


@pytest.mark.flaky(3)
def test_gaussian_constraint_sampling():
    param1 = zfit.Parameter("Param1", 5)
    params = [param1]

    mu = [5]
    sigma = [1]
    constr = GaussianConstraint(params=params, mu=mu, sigma=sigma)

    sample = constr.sample(15000)

    assert np.mean(sample[param1]) == pytest.approx(mu[0], rel=0.01)
    assert np.std(sample[param1]) == pytest.approx(sigma[0], rel=0.01)


def test_simple_constraint():

    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    mu = [3., 6.1]
    sigma = [1., 0.5]

    def func():
        return true_nll_gaussian(params=params, mu=mu, sigma=sigma)
    constr = SimpleConstraint(func=func)

    constr_np = zfit.run(constr.value())
    assert constr_np == pytest.approx(2.02)

    assert constr.get_dependents() == set(params)

#  Copyright (c) 2020 zfit
import numpy as np
import pytest
import scipy.stats

import zfit
from zfit import z
from zfit.core.constraint import BaseConstraint, SimpleConstraint, GaussianConstraint
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.container import convert_to_container
from zfit.util.exception import ShapeIncompatibleError


def true_nll_gaussian(x, mu, sigma):
    x = convert_to_container(x, container=tuple)
    mu = convert_to_container(mu, container=tuple)
    sigma = convert_to_container(sigma, container=tuple)
    constraint = z.constant(0.)
    if not len(x) == len(mu) == len(sigma):
        raise ValueError("params, mu and sigma have to have the same length.")
    for x_, mean, sig in zip(x, mu, sigma):
        constraint += z.reduce_sum(z.square(x_ - mean) / (2. * z.square(sig)))

    return constraint


def true_gauss_constr_value(x, mu, sigma):
    logpdf = lambda x, loc, scale: scipy.stats.norm.logpdf(x, loc=loc, scale=scale)
    return -np.sum(logpdf(x_, loc=mu, scale=sigma) for x_, mu, sigma in zip(x, mu, sigma))


def true_multinormal_constr_value(x, mean, cov):
    return -scipy.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)


def test_base_constraint():  # TODO(Mayou36): upgrade to tf2, use ABC again
    with pytest.raises(TypeError):
        BaseConstraint()


def test_gaussian_constraint_shape_errors():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)

    obs1 = zfit.Parameter("obs1", 2)
    obs2 = zfit.Parameter("obs2", 3)
    obs3 = zfit.Parameter("obs3", 4)

    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(params=[param1, param2], observation=[obs1, obs2, obs3], uncertainty=5).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(params=[param1, param2], observation=[obs1, obs3], uncertainty=5).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(params=[param1, param2], observation=obs1, uncertainty=[1, 4]).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(params=param1, observation=[obs1, obs3], uncertainty=[2, 3]).value()



def test_gaussian_constraint_matrix():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    observed = [3., 6.1]
    sigma = np.array([[1, 0.3],
                      [0.3, 0.5]])

    trueval = true_multinormal_constr_value(x=zfit.run(params)[0], mean=observed, cov=sigma)

    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)
    constr_np = zfit.run(constr.value())
    assert constr_np == pytest.approx(trueval)
    #assert constr_np == pytest.approx(3.989638)

    assert constr.get_dependents() == set(params)


def test_gaussian_constraint():
    param_vals = [5, 6, 3]
    observed = [3, 6.1, 4.3]
    sigma = [1, 0.3, 0.7]
    true_val = true_gauss_constr_value(x=observed, mu=param_vals, sigma=sigma)
    assert true_val == true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    params = [zfit.Parameter(f"Param{i}", val) for i, val in enumerate(param_vals)]

    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)
    constr_np = constr.value().numpy()
    assert constr_np == pytest.approx(true_val)
    assert constr.get_dependents() == set(params)

    param_vals[0] = 2
    params[0].set_value(param_vals[0])

    constr2_np = constr.value().numpy()
    constr2_newtensor_np = constr.value().numpy()
    assert constr2_newtensor_np == pytest.approx(constr2_np)

    true_val2 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    assert constr2_np == pytest.approx(true_val2)
    print(true_val2)

    constr.observation[0].set_value(5)
    observed[0] = 5
    print("x: ", param_vals, [p.numpy() for p in params])
    print("mu: ", observed, [p.numpy() for p in constr.observation])
    print("sigma: ", sigma, np.sqrt([p for p in np.diag(constr.covariance)]))
    true_val3 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    constr3_np = constr.value().numpy()
    assert constr3_np == pytest.approx(true_val3)


def test_gaussian_constraint_orderbug():  # as raised in #162
    observed = [1500, 1.0, 1.0, 1.0, 0.5]
    params = [zfit.Parameter(f"param{i}", val) for i, val in enumerate(observed)]

    sigma = [0.05 * 1500, 0.001, 0.01, 0.1, 0.05 * 0.5]
    true_val = true_gauss_constr_value(x=observed, mu=observed, sigma=sigma)

    constr1 = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)

    value_tensor = constr1.value()
    constr_np = value_tensor.numpy()
    assert constr_np == pytest.approx(true_val)
    assert true_val < 10000


def test_gaussian_constraint_orderbug2():  # as raised in #162, failed before fixing
    param1 = zfit.Parameter("param1", 1500)
    param5 = zfit.Parameter("param2", 0.5)

    param2 = zfit.Parameter("param3", 1.0)
    param3 = zfit.Parameter("param4", 1.0)
    param4 = zfit.Parameter("param5", 1.0)

    constraint = {"params": [param1, param2, param3, param4, param5],
                  "observation": [1500, 1.0, 1.0, 1.0, 0.5],
                  "uncertainty": [0.05 * 1500, 0.001, 0.01, 0.1, 0.05 * 0.5]}

    constr1 = GaussianConstraint(**constraint)
    # param_vals = [1500, 1.0, 1.0, 1.0, 0.5]
    constraint['x'] = [m.numpy() for m in constraint['params']]

    true_val = true_gauss_constr_value(x=constraint['x'], mu=constraint['observation'],
                                       sigma=constraint['uncertainty'])

    value_tensor = constr1.value()
    constr_np = value_tensor.numpy()
    assert constr_np == pytest.approx(true_val)
    assert true_val < 1000
    assert true_val == pytest.approx(-8.592, abs=0.1)  # if failing, change value. Hardcoded for additional layer


@pytest.mark.flaky(3)
def test_gaussian_constraint_sampling():
    param1 = zfit.Parameter("Param1", 5)
    params = [param1]

    observed = [5]
    sigma = [1]
    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)

    sample = constr.sample(15000)

    print(sample)

    assert np.mean(sample[constr.observation[0]]) == pytest.approx(observed[0], rel=0.01)
    assert np.std(sample[constr.observation[0]]) == pytest.approx(sigma[0], rel=0.01)


def test_simple_constraint():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    observed = [3., 6.1]
    sigma = [1., 0.5]

    def func():
        return true_nll_gaussian(x=observed, mu=params, sigma=sigma)

    constr = SimpleConstraint(func=func, params=params)

    constr_np = constr.value().numpy()
    assert constr_np == pytest.approx(2.02)

    assert constr.get_dependents() == set(params)

#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import scipy.stats

import zfit
from zfit import z
from zfit.core.constraint import BaseConstraint, GaussianConstraint, SimpleConstraint, _preprocess_gaussian_constr_sigma_var
import zfit.z.numpy as znp
from zfit.core.constraint import BaseConstraint, GaussianConstraint, SimpleConstraint
from zfit.util.container import convert_to_container
from zfit.util.exception import ShapeIncompatibleError


def true_nll_gaussian(x, mu, sigma):
    x = convert_to_container(x, container=tuple)
    mu = convert_to_container(mu, container=tuple)
    sigma = convert_to_container(sigma, container=tuple)
    constraint = z.constant(0.0)
    if not len(x) == len(mu) == len(sigma):
        raise ValueError("params, mu and sigma have to have the same length.")
    for x_, mean, sig in zip(x, mu, sigma):
        constraint += z.reduce_sum(z.square(x_ - mean) / (2.0 * z.square(sig)))

    return constraint


def true_gauss_constr_value(x, mu, sigma):
    logpdf = lambda x, loc, scale: scipy.stats.norm.logpdf(x, loc=loc, scale=scale)
    return -np.sum(
        [logpdf(x_, loc=mu, scale=sigma) for x_, mu, sigma in zip(x, mu, sigma)]
    )


def true_poisson_constr_value(x, lam):
    logpdf = lambda x, loc: scipy.stats.poisson.logpmf(x, mu=loc)
    return -np.sum([logpdf(x_, loc=lam_) for x_, lam_ in zip(x, lam)])


def true_multinormal_constr_value(x, mean, cov):
    return -scipy.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)


def test_base_constraint():
    with pytest.raises(TypeError):
        BaseConstraint()


def test_poisson_constrain():
    x, lam = np.random.randint(1, 100, size=(2, 50))
    constr = zfit.constraint.PoissonConstraint(
        params=z.convert_to_tensor(x), observation=z.convert_to_tensor(lam)
    )
    poiss_constr_val = constr.value()
    true_val = true_poisson_constr_value(x, lam)
    np.testing.assert_allclose(poiss_constr_val, true_val)


def test_gaussian_constraint_shape_errors():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)

    obs1 = zfit.Parameter("obs1", 2)
    obs2 = zfit.Parameter("obs2", 3)
    obs3 = zfit.Parameter("obs3", 4)

    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(
            params=[param1, param2], observation=[obs1, obs2, obs3], uncertainty=5
        ).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(
            params=[param1, param2], observation=[obs1, obs3], uncertainty=5
        ).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(
            params=[param1, param2], observation=obs1, uncertainty=[1, 4]
        ).value()
    with pytest.raises(ShapeIncompatibleError):
        GaussianConstraint(
            params=param1, observation=[obs1, obs3], uncertainty=[2, 3]
        ).value()


def test_gaussian_constraint_matrix_legacy():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    observed = [3.0, 6.1]
    sigma = np.array([[1, 0.3], [0.3, 0.5]])

    trueval = true_multinormal_constr_value(
        x=znp.asarray(params), mean=observed, cov=sigma
    )

    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)
    constr_np = znp.asarray(constr.value())
    assert pytest.approx(trueval) == constr_np
    # assert constr_np == pytest.approx(3.989638)

    assert set(constr.get_params()) == set(params)


@pytest.mark.parametrize("kwargs", [{'sigma': np.array([[1, 0.3], [0.3, 0.5]]) ** 0.5}, {'cov': np.array([[1, 0.3], [0.3, 0.5]])}], ids=['sigma', 'cov'])
def test_gaussian_constraint_matrix(kwargs):
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    observed = [3.0, 6.1]
    sigma = np.array([[1, 0.3], [0.3, 0.5]])

    trueval = true_multinormal_constr_value(
        x=zfit.run(params), mean=observed, cov=sigma
    )
    if 'sigma' in kwargs:
        with pytest.raises(ValueError):
            _ = GaussianConstraint(params=params, observation=observed, **kwargs)
        return
    constr = GaussianConstraint(params=params, observation=observed, **kwargs)
    constr_np = zfit.run(constr.value())
    assert constr_np == pytest.approx(trueval)
    # assert constr_np == pytest.approx(3.989638)

    assert set(constr.get_params()) == set(params)



def test_gausian_constraint_matrix_preprocess():

    covin = np.array([[1, 0.3], [0.3, 0.5]])
    sigma, cov = _preprocess_gaussian_constr_sigma_var(covin, None, False)
    np.testing.assert_allclose(sigma, np.sqrt([covin[0, 0], covin[1, 1]]))
    np.testing.assert_allclose(cov, covin)

    covin = np.array([2.0, 0.3])
    sigma, cov = _preprocess_gaussian_constr_sigma_var(covin, None, False)
    np.testing.assert_allclose(sigma ** 2, covin)
    np.testing.assert_allclose(cov, np.array([[covin[0], 0], [0, covin[1]]]))

    covin = np.array([9.])
    sigma, cov = _preprocess_gaussian_constr_sigma_var(covin, None, False)
    np.testing.assert_allclose(sigma ** 2, covin)
    np.testing.assert_allclose(cov, np.array([[covin[0]]]))

    covin = 8.
    sigma, cov = _preprocess_gaussian_constr_sigma_var(covin, None, False)
    np.testing.assert_allclose(sigma ** 2, np.array([covin]))
    np.testing.assert_allclose(cov, np.array([[covin]]))

    covin = [1, 0.3]
    sigma, cov = _preprocess_gaussian_constr_sigma_var(covin, None, False)
    np.testing.assert_allclose(sigma, np.sqrt(covin))
    np.testing.assert_allclose(cov, np.diag(covin))

    sigmain = np.array([1, 0.3])
    sigma, cov = _preprocess_gaussian_constr_sigma_var(None, sigmain, False)
    np.testing.assert_allclose(sigma, sigmain)
    np.testing.assert_allclose(cov, np.array([[sigmain[0], 0], [0, sigmain[1]]]) ** 2)

    sigmain = 3.
    sigma, cov = _preprocess_gaussian_constr_sigma_var(None, sigmain, False)
    np.testing.assert_allclose(sigma, np.array([sigmain]))
    np.testing.assert_allclose(cov, np.array([[sigmain]]) ** 2)





def test_gaussian_constraint_legacy():
    param_vals = [5, 6, 3]
    observed = [zfit.Parameter(f"observed {val}", val) for val in [3, 6.1, 4.3]]
    sigma = [1, 0.3, 0.7]
    true_val = true_gauss_constr_value(x=observed, mu=param_vals, sigma=sigma)
    assert true_val == true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    params = [zfit.Parameter(f"Param{i}", val) for i, val in enumerate(param_vals)]

    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)
    constr_np = constr.value()
    assert pytest.approx(true_val) == constr_np
    assert set(constr.get_params()) == set(params)

    param_vals[0] = 2
    params[0].set_value(param_vals[0])

    constr2_np = constr.value()
    constr2_newtensor_np = constr.value()
    assert pytest.approx(constr2_np) == constr2_newtensor_np

    true_val2 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    assert pytest.approx(true_val2) == constr2_np

    constr.observation[0].set_value(5)
    observed[0] = 5
    # print("x: ", param_vals, [p for p in params])
    # print("mu: ", observed, [p for p in constr.observation])
    # print("sigma: ", sigma, np.sqrt([p for p in np.diag(constr.covariance)]))
    true_val3 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    constr3_np = constr.value()
    assert pytest.approx(true_val3) == constr3_np

@pytest.mark.parametrize("kwargs", [{'sigma': [1, 0.3, 0.7]}, {'cov': np.array([1, 0.3, 0.7]) ** 2},
                                    {'cov': np.array([[1, 0, 0], [0, 0.3, 0], [0, 0, 0.7]]) ** 2}],
                         ids=['sigma', 'cov', 'cov_matrix'])
def test_gaussian_constraint(kwargs):
    param_vals = [5, 6, 3]
    observed = [zfit.Parameter(f"observed {val}", val) for val in [3, 6.1, 4.3]]
    sigma = [1, 0.3, 0.7]
    true_val = true_gauss_constr_value(x=observed, mu=param_vals, sigma=sigma)
    assert true_val == true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    params = [zfit.Parameter(f"Param{i}", val) for i, val in enumerate(param_vals)]

    constr = GaussianConstraint(params=params, observation=observed, **kwargs)
    constr_np = constr.value().numpy()
    assert constr_np == pytest.approx(true_val)
    assert set(constr.get_params()) == set(params)

    param_vals[0] = 2
    params[0].set_value(param_vals[0])

    constr2_np = constr.value().numpy()
    constr2_newtensor_np = constr.value().numpy()
    assert constr2_newtensor_np == pytest.approx(constr2_np)

    true_val2 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    assert constr2_np == pytest.approx(true_val2)

    constr.observation[0].set_value(5)
    observed[0] = 5
    true_val3 = true_gauss_constr_value(x=param_vals, mu=observed, sigma=sigma)
    constr3_np = constr.value().numpy()
    assert constr3_np == pytest.approx(true_val3)


sigma_true_orderbug = [0.05 * 1500, 0.001, 0.01, 0.1, 0.05 * 0.5]


@pytest.mark.parametrize("kwargs", [{'sigma': sigma_true_orderbug},
                                    {'cov': np.diag(sigma_true_orderbug) ** 2}],
                            ids=['sigma', 'cov'])
def test_gaussian_constraint_orderbug(kwargs):  # as raised in #162
    observed = [1500, 1.0, 1.0, 1.0, 0.5]
    params = [zfit.Parameter(f"param{i}", val) for i, val in enumerate(observed)]

    sigma = sigma_true_orderbug
    true_val = true_gauss_constr_value(x=observed, mu=observed, sigma=sigma)

    constr1 = GaussianConstraint(params=params, observation=observed, **kwargs)

    value_tensor = constr1.value()
    constr_np = value_tensor.numpy()
    assert constr_np == pytest.approx(true_val)
    assert true_val < 10000

def test_gaussian_constraint_orderbug_legacy():  # as raised in #162
    observed = [1500, 1.0, 1.0, 1.0, 0.5]
    params = [zfit.Parameter(f"param{i}", val) for i, val in enumerate(observed)]

    sigma = sigma_true_orderbug
    true_val = true_gauss_constr_value(x=observed, mu=observed, sigma=sigma)

    constr1 = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)

    assert pytest.approx(true_val) == constr1.value()
    assert true_val < 10000


def test_gaussian_constraint_orderbug2_legacy():  # as raised in #162, failed before fixing
    param1 = zfit.Parameter("param1", 1500)
    param5 = zfit.Parameter("param2", 0.5)

    param2 = zfit.Parameter("param3", 1.0)
    param3 = zfit.Parameter("param4", 1.0)
    param4 = zfit.Parameter("param5", 1.0)

    constraint = {
        "params": [param1, param2, param3, param4, param5],
        "observation": [1500, 1.0, 1.0, 1.0, 0.5],
        "uncertainty": sigma_true_orderbug,
    }

    constr1 = GaussianConstraint(**constraint)
    # param_vals = [1500, 1.0, 1.0, 1.0, 0.5]
    constraint["x"] = constraint["params"]

    true_val = true_gauss_constr_value(
        x=constraint["x"], mu=constraint["observation"], sigma=constraint["uncertainty"]
    )

    assert pytest.approx(true_val) == constr1.value()
    assert true_val < 1000
    assert pytest.approx(
        -8.592, abs=0.1
    ) == true_val  # if failing, change value. Hardcoded for additional layer


@pytest.mark.parametrize("kwargs", [{'sigma': sigma_true_orderbug},
                                    {'cov': np.diag(sigma_true_orderbug) ** 2},
                                    {'cov': np.array(sigma_true_orderbug) ** 2}],                            ids=['sigma', 'cov_matrix', 'cov'])
def test_gaussian_constraint_orderbug2(kwargs):  # as raised in #162, failed before fixing
    param1 = zfit.Parameter("param1", 1500)
    param5 = zfit.Parameter("param2", 0.5)

    param2 = zfit.Parameter("param3", 1.0)
    param3 = zfit.Parameter("param4", 1.0)
    param4 = zfit.Parameter("param5", 1.0)

    constraint = {
        "params": [param1, param2, param3, param4, param5],
        "observation": [1500, 1.0, 1.0, 1.0, 0.5],
        **kwargs
    }

    constr1 = GaussianConstraint(**constraint)
    # param_vals = [1500, 1.0, 1.0, 1.0, 0.5]
    constraint["x"] = [m.numpy() for m in constraint["params"]]

    true_val = true_gauss_constr_value(
        x=constraint["x"], mu=constraint["observation"], sigma=sigma_true_orderbug
    )

    value_tensor = constr1.value()
    constr_np = value_tensor.numpy()
    assert constr_np == pytest.approx(true_val)
    assert true_val < 1000
    assert true_val == pytest.approx(
        -8.592, abs=0.1
    )  # if failing, change value. Hardcoded for additional layer


@pytest.mark.flaky(3)
@pytest.mark.parametrize("kwargs", [{'sigma': [1, 0.3, 0.7]}, {'cov': np.array([1, 0.3, 0.7]) ** 2},
                                    {'cov': np.array([[1, 0, 0], [0, 0.3, 0], [0, 0, 0.7]]) ** 2}],                  ids=['sigma', 'cov', 'cov_matrix'])
def test_gaussian_constraint_sampling(kwargs):
    param1 = zfit.Parameter("Param1", 5)
    params = [param1]

    observed = [5]
    sigma = [1]
    constr = GaussianConstraint(params=params, observation=observed, **kwargs)

    sample = constr.sample(15000)

    assert np.mean(sample[param1]) == pytest.approx(observed[0], rel=0.01)
    assert np.std(sample[param1]) == pytest.approx(sigma[0], rel=0.01)

@pytest.mark.flaky(3)
def test_gaussian_constraint_sampling_legacy():
    param1 = zfit.Parameter("Param1", 5)
    params = [param1]

    observed = [5]
    sigma = [1]
    constr = GaussianConstraint(params=params, observation=observed, uncertainty=sigma)

    sample = constr.sample(15000)

    assert pytest.approx(observed[0], rel=0.01) == np.mean(sample[param1])
    assert pytest.approx(sigma[0], rel=0.01) == np.std(sample[param1])


def test_simple_constraint_legacy():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    observed = [3.0, 6.1]
    sigma = [1.0, 0.5]

    def func():
        return true_nll_gaussian(x=observed, mu=params, sigma=sigma)

    constr = SimpleConstraint(func=func, params=params)

    constr_np = constr.value()
    assert pytest.approx(2.02) == constr_np

    assert set(constr.get_params()) == set(params)

def test_gauss_fails_params():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]
    sigma = [1.0, param1]
    observed = [3.0, 6.1]
    with pytest.raises(ValueError):
        GaussianConstraint(params=params, observation=observed, sigma=sigma)
    cov = [2.0, param2]
    with pytest.raises(ValueError):
        GaussianConstraint(params=params, observation=observed, cov=cov)


def test_simple_constraint_paramfunc():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = {"p1": param1, "p2": param2}

    observed = [3.0, 6.1]
    sigma = [1.0, 0.5]

    def func(params):
        return true_nll_gaussian(
            x=observed, mu=[params["p1"], params["p2"]], sigma=sigma
        )

    constr = SimpleConstraint(func=func, params=params)

    assert pytest.approx(2.02) == constr.value()

    assert set(constr.get_params()) == set(params.values())


def test_log_normal_constraint():
    # x, lam = np.random.randint(1, 100, size=(2, 50))
    lam = 44.3
    x = 45.3
    lam_tensor = z.convert_to_tensor(lam)
    constr = zfit.constraint.LogNormalConstraint(
        params=z.convert_to_tensor(x),
        observation=lam_tensor,
        uncertainty=lam_tensor**0.5,
    )
    lognormal_constr_val = constr.value()
    # true_val = true_poisson_constr_value(x, lam)
    true_lognormal = 25.128554  # maybe add dynamically?
    np.testing.assert_allclose(lognormal_constr_val, true_lognormal)

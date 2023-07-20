#  Copyright (c) 2023 zfit
import numpy as np
import pytest
import tensorflow as tf

import zfit.core.basepdf
import zfit.models.dist_tfp
import zfit.settings
from zfit import z
from zfit.core.loss import UnbinnedNLL
from zfit.core.space import Space
from zfit.minimize import Minuit
from zfit.pdf import Gauss
from zfit.util.exception import BehaviorUnderDiscussion, IntentionAmbiguousError

mu_true = 1.2
sigma_true = 4.1
mu_true2 = 1.01
sigma_true2 = 3.5

yield_true = 3000
test_values_np = np.random.normal(loc=mu_true, scale=sigma_true, size=(yield_true, 1))


def create_test_values(size):
    return z.random.get_prng().normal(mean=mu_true, stddev=sigma_true, shape=(size, 1))


test_values_np2 = np.random.normal(loc=mu_true2, scale=sigma_true2, size=yield_true)

low, high = -24.3, 28.6


def create_params1(nameadd=""):
    mu1 = zfit.Parameter(
        "mu1" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1.0, mu_true + 1.0
    )
    sigma1 = zfit.Parameter(
        "sigma1" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 2.0,
        sigma_true + 2.0,
    )
    return mu1, sigma1


def create_params2(nameadd=""):
    mu2 = zfit.Parameter(
        "mu25" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1.0, mu_true + 1.0
    )
    sigma2 = zfit.Parameter(
        "sigma25" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 2.0,
        sigma_true + 2.0,
    )
    return mu2, sigma2


def create_params3(nameadd=""):
    mu3 = zfit.Parameter(
        "mu35" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1.0, mu_true + 1.0
    )
    sigma3 = zfit.Parameter(
        "sigma35" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 2.0,
        sigma_true + 2.0,
    )
    yield3 = zfit.Parameter("yield35" + nameadd, yield_true + 300, 0, 10000000)
    return mu3, sigma3, yield3


obs1 = zfit.Space(
    "obs1",
    (
        np.min([test_values_np[:, 0], test_values_np2]) - 1.4,
        np.max([test_values_np[:, 0], test_values_np2]) + 2.4,
    ),
)

mu_constr = [1.6, 0.02]  # mu, sigma
sigma_constr = [3.5, 0.01]
constr = lambda: [mu_constr[1], sigma_constr[1]]
constr_tf = lambda: z.convert_to_tensor(constr())
covariance = lambda: np.array([[mu_constr[1] ** 2, 0], [0, sigma_constr[1] ** 2]])
covariance_tf = lambda: z.convert_to_tensor(covariance())


def create_gauss1(obs=obs1):
    mu, sigma = create_params1()
    return Gauss(mu, sigma, obs=obs, name="gaussian1"), mu, sigma


def create_gauss2(obs=obs1):
    mu, sigma = create_params2()
    return Gauss(mu, sigma, obs=obs, name="gaussian2"), mu, sigma


def create_gauss3ext():
    mu, sigma, yield3 = create_params3()
    gaussian3 = Gauss(mu, sigma, obs=obs1, name="gaussian3")
    gaussian3 = gaussian3.create_extended(yield3)
    return gaussian3, mu, sigma, yield3


def create_simpel_loss():
    _, mu1, sigma1 = create_gauss1()
    return zfit.loss.SimpleLoss(
        lambda x: (x[0] - 0.37) ** 4 * (x[1] - 2.34) ** 4,
        params=[mu1, sigma1],
        errordef=0.5,
    )


def create_simultaneous_loss():
    test_values = tf.constant(test_values_np)
    test_values = zfit.Data.from_tensor(obs=obs1, tensor=test_values)
    test_values2 = tf.constant(test_values_np2)
    test_values2 = zfit.Data.from_tensor(obs=obs1, tensor=test_values2)
    gaussian1, mu1, sigma1 = create_gauss1()
    gaussian2, mu2, sigma2 = create_gauss2()
    gaussian2 = gaussian2.create_extended(zfit.Parameter("yield_gauss2", 5))
    nll = zfit.loss.UnbinnedNLL(
        model=[gaussian1, gaussian2],
        data=[test_values, test_values2],
    )
    return mu1, mu2, nll, sigma1, sigma2


@pytest.mark.parametrize("size", [None, 5000, 50000, 300000, 3_000_000])
@pytest.mark.flaky(3)  # minimization can fail
def test_extended_unbinned_nll(size):
    if size is None:
        test_values = z.constant(test_values_np)
        size = test_values.shape[0]
    else:
        test_values = create_test_values(size)
    test_values = zfit.Data.from_tensor(obs=obs1, tensor=test_values)
    gaussian3, mu3, sigma3, yield3 = create_gauss3ext()
    nll = zfit.loss.ExtendedUnbinnedNLL(
        model=gaussian3, data=test_values, fit_range=(-20, 20)
    )
    assert {mu3, sigma3, yield3} == nll.get_params()
    minimizer = Minuit(tol=1e-4)
    status = minimizer.minimize(loss=nll)
    params = status.params
    assert params[mu3]["value"] == pytest.approx(
        zfit.run(tf.math.reduce_mean(test_values.value())), rel=0.05
    )
    assert params[sigma3]["value"] == pytest.approx(
        zfit.run(tf.math.reduce_std(test_values.value())), rel=0.05
    )
    assert params[yield3]["value"] == pytest.approx(size, rel=0.005)


def test_unbinned_simultaneous_nll():
    mu1, mu2, nll, sigma1, sigma2 = create_simultaneous_loss()
    minimizer = Minuit(tol=1e-5)
    status = minimizer.minimize(loss=nll, params=[mu1, sigma1, mu2, sigma2])
    params = status.params
    assert set(nll.get_params()) == {mu1, mu2, sigma1, sigma2}

    assert params[mu1]["value"] == pytest.approx(np.mean(test_values_np), rel=0.007)
    assert params[mu2]["value"] == pytest.approx(np.mean(test_values_np2), rel=0.007)
    assert params[sigma1]["value"] == pytest.approx(np.std(test_values_np), rel=0.007)
    assert params[sigma2]["value"] == pytest.approx(np.std(test_values_np2), rel=0.007)


@pytest.mark.flaky(3)
@pytest.mark.parametrize(
    "weights",
    (None, np.random.normal(loc=1.0, scale=0.2, size=test_values_np.shape[0])),
)
@pytest.mark.parametrize("sigma", (constr, constr_tf, covariance, covariance_tf))
@pytest.mark.parametrize("options", ({"subtr_const": False}, {"subtr_const": True}))
def test_unbinned_nll(weights, sigma, options):
    gaussian1, mu1, sigma1 = create_gauss1()
    gaussian2, mu2, sigma2 = create_gauss2()

    test_values = tf.constant(test_values_np)
    test_values = zfit.Data.from_tensor(obs=obs1, tensor=test_values, weights=weights)
    nll_object = zfit.loss.UnbinnedNLL(
        model=gaussian1, data=test_values, options=options
    )
    minimizer = Minuit(tol=1e-5)
    status = minimizer.minimize(loss=nll_object, params=[mu1, sigma1])
    params = status.params
    rel_error = 0.12 if weights is None else 0.1  # more fluctuating with weights

    assert params[mu1]["value"] == pytest.approx(np.mean(test_values_np), rel=rel_error)
    assert params[sigma1]["value"] == pytest.approx(
        np.std(test_values_np), rel=rel_error
    )

    constraints = zfit.constraint.nll_gaussian(
        params=[mu2, sigma2],
        observation=[mu_constr[0], sigma_constr[0]],
        uncertainty=sigma(),
    )
    nll_object = UnbinnedNLL(
        model=gaussian2, data=test_values, constraints=constraints, options=options
    )

    minimizer = Minuit(tol=1e-4)
    status = minimizer.minimize(loss=nll_object, params=[mu2, sigma2])
    params = status.params
    if weights is None:
        assert params[mu2]["value"] > np.average(test_values_np, weights=weights)
        assert params[sigma2]["value"] < np.std(test_values_np)


def test_add():
    param1 = zfit.Parameter("param1", 1.0)
    param2 = zfit.Parameter("param2", 2.0)
    param3 = zfit.Parameter("param3", 2.0)

    pdfs = [0] * 4
    pdfs[0] = Gauss(param1, 4, obs=obs1)
    pdfs[1] = Gauss(param2, 5, obs=obs1)
    pdfs[2] = Gauss(3, 6, obs=obs1)
    pdfs[3] = Gauss(4, 7, obs=obs1)

    datas = [0] * 4
    datas[0] = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(1.0))
    datas[1] = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(2.0))
    datas[2] = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(3.0))
    datas[3] = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(4.0))

    ranges = [0] * 4
    ranges[0] = (1, 4)
    ranges[1] = Space(limits=(2, 5), obs=obs1)
    ranges[2] = Space(limits=(3, 6), obs=obs1)
    ranges[3] = Space(limits=(4, 7), obs=obs1)

    constraint1 = zfit.constraint.nll_gaussian(
        params=param1, observation=1.0, uncertainty=0.5
    )
    constraint2 = zfit.constraint.nll_gaussian(
        params=param3, observation=2.0, uncertainty=0.25
    )
    merged_contraints = [constraint1, constraint2]

    nll1 = UnbinnedNLL(
        model=pdfs[0], data=datas[0], fit_range=ranges[0], constraints=constraint1
    )
    nll2 = UnbinnedNLL(
        model=pdfs[1], data=datas[1], fit_range=ranges[1], constraints=constraint2
    )
    nll3 = UnbinnedNLL(
        model=[pdfs[2], pdfs[3]],
        data=[datas[2], datas[3]],
        fit_range=[ranges[2], ranges[3]],
    )

    simult_nll = nll1 + nll2 + nll3

    assert simult_nll.model == pdfs
    assert simult_nll.data == datas

    ranges[0] = Space(
        limits=ranges[0], obs="obs1", axes=(0,)
    )  # for comparison, Space can only compare with Space
    ranges[1].coords._axes = (0,)
    ranges[2].coords._axes = (0,)
    ranges[3].coords._axes = (0,)
    assert simult_nll.fit_range == ranges

    def eval_constraint(constraints):
        return z.reduce_sum([c.value() for c in constraints]).numpy()

    assert eval_constraint(simult_nll.constraints) == eval_constraint(merged_contraints)
    assert set(simult_nll.get_params()) == {param1, param2, param3}


@pytest.mark.parametrize("chunksize", [10000000, 1000])
def test_gradients(chunksize):
    from numdifftools import Gradient

    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize

    initial1 = 1.0
    initial2 = 2
    param1 = zfit.Parameter("param1", initial1)
    param2 = zfit.Parameter("param2", initial2)

    gauss1 = Gauss(param1, 4, obs=obs1)
    gauss1.set_norm_range((-5, 5))
    gauss2 = Gauss(param2, 5, obs=obs1)
    gauss2.set_norm_range((-5, 5))

    data1 = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(1.0, shape=(100,)))
    data1.set_data_range((-5, 5))
    data2 = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(1.0, shape=(100,)))
    data2.set_data_range((-5, 5))

    nll = UnbinnedNLL(model=[gauss1, gauss2], data=[data1, data2])

    def loss_func(values):
        for val, param in zip(values, nll.get_cache_deps(only_floating=True)):
            param.set_value(val)
        return nll.value().numpy()

    # theoretical, numerical = tf.test.compute_gradient(loss_func, list(params))
    gradient1 = nll.gradient(params=param1)
    gradient_func = Gradient(loss_func)
    # gradient_func = lambda *args, **kwargs: list(gradient_func_numpy(*args, **kwargs))
    assert gradient1[0].numpy() == pytest.approx(gradient_func([param1.numpy()]))
    param1.set_value(initial1)
    param2.set_value(initial2)
    params = [param2, param1]
    gradient2 = nll.gradient(params=params)
    both_gradients_true = list(
        reversed(list(gradient_func([initial1, initial2])))
    )  # because param2, then param1
    assert [g.numpy() for g in gradient2] == pytest.approx(both_gradients_true)

    param1.set_value(initial1)
    param2.set_value(initial2)
    gradient3 = nll.gradient()
    gradients_true3 = []
    for param_o in nll.get_params():
        for param, grad in zip(params, gradient2):
            if param_o is param:
                gradients_true3.append(float(grad))
                break
    assert [g.numpy() for g in gradient3] == pytest.approx(gradients_true3)


def test_simple_loss():
    true_a = 1.0
    true_b = 4.0
    true_c = -0.3
    truevals = true_a, true_b, true_c
    a_param = zfit.Parameter(
        "variable_a15151loss", 1.5, -1.0, 20.0, step_size=z.constant(0.1)
    )
    b_param = zfit.Parameter("variable_b15151loss", 3.5)
    c_param = zfit.Parameter("variable_c15151loss", -0.23)
    param_list = [a_param, b_param, c_param]

    def loss_func(params):
        a_param, b_param, c_param = params
        probs = (
            z.convert_to_tensor(
                (a_param - true_a) ** 2
                + (b_param - true_b) ** 2
                + (c_param - true_c) ** 4
            )
            + 0.42
        )
        return tf.reduce_sum(input_tensor=tf.math.log(probs))

    with pytest.raises(ValueError):
        _ = zfit.loss.SimpleLoss(func=loss_func, params=param_list)

    loss_func.errordef = 1
    loss_deps = zfit.loss.SimpleLoss(func=loss_func, params=param_list)
    loss = zfit.loss.SimpleLoss(func=loss_func, params=param_list)
    loss2 = zfit.loss.SimpleLoss(func=loss_func, params=truevals)

    assert loss_deps.get_cache_deps() == set(param_list)
    assert set(loss_deps.get_params()) == set(param_list)

    loss_tensor = loss_func(param_list)
    loss_value_np = loss_tensor.numpy()

    assert loss.value().numpy() == pytest.approx(loss_value_np)
    assert loss_deps.value().numpy() == pytest.approx(loss_value_np)

    assert loss.value(full=True).numpy() == pytest.approx(
        loss_deps.value(full=True).numpy()
    )

    with pytest.raises(IntentionAmbiguousError):
        _ = loss + loss_deps

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=loss)
    assert result.valid
    assert true_a == pytest.approx(result.params[a_param]["value"], rel=0.03)
    assert true_b == pytest.approx(result.params[b_param]["value"], rel=0.06)
    assert true_c == pytest.approx(result.params[c_param]["value"], rel=0.5)

    zfit.param.set_values(param_list, np.array(zfit.run(param_list)) + 0.6)
    result2 = minimizer.minimize(loss=loss2)
    assert result2.valid
    params = list(result2.params)
    assert true_a == pytest.approx(result2.params[params[0]]["value"], rel=0.03)
    assert true_b == pytest.approx(result2.params[params[1]]["value"], rel=0.06)
    assert true_c == pytest.approx(result2.params[params[2]]["value"], rel=0.5)


def test_create_new_nll():
    gaussian1, mu1, sigma1 = create_gauss1()
    gaussian2, mu2, sigma2 = create_gauss2()

    test_values = tf.constant(test_values_np)
    test_values = zfit.Data.from_tensor(obs=obs1, tensor=test_values)
    nll = zfit.loss.UnbinnedNLL(model=gaussian1, data=test_values)

    nll2 = nll.create_new(model=gaussian2)
    assert nll2.data[0] is nll.data[0]
    assert nll2.constraints == nll.constraints
    assert nll2._options == nll._options

    nll3 = nll.create_new()
    assert nll3.data[0] is nll.data[0]
    assert nll3.constraints == nll.constraints
    assert nll3._options == nll._options

    nll4 = nll.create_new(options={})
    assert nll4.data[0] is nll.data[0]
    assert nll4.constraints == nll.constraints
    assert nll4._options != nll._constraints


def test_create_new_extnll():
    gaussian1, mu1, sigma1, yield1 = create_gauss3ext()

    test_values = tf.constant(test_values_np)
    test_values = zfit.Data.from_tensor(obs=obs1, tensor=test_values)
    nll = zfit.loss.ExtendedUnbinnedNLL(
        model=gaussian1,
        data=test_values,
        constraints=zfit.constraint.GaussianConstraint(mu1, 1.0, 0.1),
    )

    nll2 = nll.create_new(model=gaussian1)
    assert nll2.data[0] is nll.data[0]
    assert nll2.constraints == nll.constraints
    assert nll2._options == nll._options

    nll3 = nll.create_new()
    assert nll3.data[0] is nll.data[0]
    assert nll3.constraints == nll.constraints
    assert nll3._options == nll._options

    nll4 = nll.create_new(options={})
    assert nll4.data[0] is nll.data[0]
    assert nll4.constraints == nll.constraints
    assert nll4._options != nll._constraints


def test_create_new_simple():
    _, mu1, sigma1 = create_gauss1()

    loss = zfit.loss.SimpleLoss(lambda x, y: x * y, params=[mu1, sigma1], errordef=0.5)

    loss1 = loss.create_new()
    assert loss1._simple_func is loss._simple_func
    assert loss1.errordef == loss.errordef


@pytest.mark.parametrize(
    "create_loss", [lambda: create_simultaneous_loss()[2], create_simpel_loss]
)
def test_callable_loss(create_loss):
    loss = create_loss()

    params = list(loss.get_params())
    x = np.array(zfit.run(params)) + 0.1
    value_loss = loss(x)
    with zfit.param.set_values(params, x):
        true_val = zfit.run(loss.value())
        _ = zfit.run(loss.value(full=True))
        assert true_val == pytest.approx(zfit.run(value_loss))
        with pytest.raises(BehaviorUnderDiscussion):
            assert true_val == pytest.approx(zfit.run(loss()))

    with pytest.raises(ValueError):
        loss(x[:-1])
    with pytest.raises(ValueError):
        loss(list(x) + [1])


@pytest.mark.parametrize(
    "create_loss", [lambda: create_simultaneous_loss()[2], create_simpel_loss]
)
def test_iminuit_compatibility(create_loss):
    loss = create_loss()

    params = list(loss.get_params())
    x = np.array(zfit.run(params)) + 0.1
    zfit.param.set_values(params, x)

    with pytest.raises(ValueError):
        loss(x[:-1])
    with pytest.raises(ValueError):
        loss(list(x) + [1])

    import iminuit

    minimizer = iminuit.Minuit(loss, x)
    result = minimizer.migrad()
    assert result.valid
    minimizer.hesse()

    zfit.param.set_values(params, x)
    minimizer_zfit = zfit.minimize.Minuit()
    result_zfit = minimizer_zfit.minimize(loss)
    assert result_zfit.fmin == pytest.approx(result.fmin.fval, abs=0.03)


@pytest.mark.flaky(3)
# @pytest.mark.parametrize('weights', [None, np.random.normal(loc=1., scale=0.2, size=test_values_np.shape[0])])
@pytest.mark.parametrize("weights", [None])
def test_binned_nll(weights):
    obs = zfit.Space("obs1", limits=(-15, 25))
    gaussian1, mu1, sigma1 = create_gauss1(obs=obs)
    gaussian2, mu2, sigma2 = create_gauss2(obs=obs)
    test_values_np = np.random.normal(loc=mu_true, scale=4, size=(10000, 1))

    test_values = tf.constant(test_values_np)
    test_values = zfit.Data.from_tensor(obs=obs, tensor=test_values, weights=weights)
    test_values_binned = test_values.to_binned(100)
    nll_object = zfit.loss.BinnedNLL(
        model=gaussian1.to_binned(test_values_binned.axes), data=test_values_binned
    )
    minimizer = Minuit()
    status = minimizer.minimize(loss=nll_object, params=[mu1, sigma1])
    params = status.params
    rel_error = 0.035 if weights is None else 0.15  # more fluctuating with weights

    assert params[mu1]["value"] == pytest.approx(np.mean(test_values_np), rel=rel_error)
    assert params[sigma1]["value"] == pytest.approx(
        np.std(test_values_np), rel=rel_error
    )

    constraints = zfit.constraint.GaussianConstraint(
        params=[mu2, sigma2],
        observation=[mu_constr[0], sigma_constr[0]],
        uncertainty=[mu_constr[1], sigma_constr[1]],
    )
    nll_object = zfit.loss.BinnedNLL(
        model=gaussian2.to_binned(test_values_binned.axes),
        data=test_values_binned,
        constraints=constraints,
    )

    minimizer = Minuit()
    status = minimizer.minimize(loss=nll_object, params=[mu2, sigma2])
    params = status.params
    if weights is None:
        assert params[mu2]["value"] > np.mean(test_values_np)
        assert params[sigma2]["value"] < np.std(test_values_np)

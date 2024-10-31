#  Copyright (c) 2024 zfit
import jacobi
import numpy as np
import pytest
import tensorflow as tf

import zfit.core.basepdf
import zfit.models.dist_tfp
import zfit.settings
from zfit import z
import zfit.z.numpy as znp
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
    assert pytest.approx(
        np.mean(test_values.value()), rel=0.05
    ) == params[mu3]["value"]
    assert pytest.approx(
        np.std(test_values.value()), rel=0.05
    ) == params[sigma3]["value"]
    assert pytest.approx(size, rel=0.005) == params[yield3]["value"]


def test_unbinned_simultaneous_nll():
    mu1, mu2, nll, sigma1, sigma2 = create_simultaneous_loss()
    minimizer = Minuit(tol=1e-5)
    status = minimizer.minimize(loss=nll, params=[mu1, sigma1, mu2, sigma2])
    params = status.params
    assert set(nll.get_params()) == {mu1, mu2, sigma1, sigma2}

    assert pytest.approx(np.mean(test_values_np), rel=0.007) == params[mu1]["value"]
    assert pytest.approx(np.mean(test_values_np2), rel=0.007) == params[mu2]["value"]
    assert pytest.approx(np.std(test_values_np), rel=0.007) == params[sigma1]["value"]
    assert pytest.approx(np.std(test_values_np2), rel=0.007) == params[sigma2]["value"]


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

    assert pytest.approx(np.mean(test_values_np), rel=rel_error) == params[mu1]["value"]
    assert pytest.approx(
        np.std(test_values_np), rel=rel_error
    ) == params[sigma1]["value"]

    constraints = zfit.constraint.GaussianConstraint(
        params=[mu2, sigma2],
        observation=[mu_constr[0], sigma_constr[0]],
        cov=sigma(),
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

    constraint1 = zfit.constraint.GaussianConstraint(
        params=param1, observation=1.0, sigma=0.5
    )
    constraint2 = zfit.constraint.GaussianConstraint(
        params=param3, observation=2.0, sigma=0.25
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
        return z.reduce_sum([c.value() for c in constraints])

    assert eval_constraint(simult_nll.constraints) == eval_constraint(merged_contraints)
    assert set(simult_nll.get_params()) == {param1, param2, param3}


@pytest.mark.parametrize("numgrad", [True, False], ids=["numgrad", "autograd"])
@pytest.mark.parametrize("chunksize", [10000000, 1000])
def test_gradients(chunksize, numgrad):
    from numdifftools import Gradient

    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize

    initial1 = 1.0
    initial2 = 2
    param1 = zfit.Parameter("param1", initial1)
    param2 = zfit.Parameter("param2", initial2)

    gauss1 = Gauss(param1, 4, obs=obs1)
    gauss2 = Gauss(param2, 5, obs=obs1)

    data1 = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(1.0, shape=(100,)))
    data2 = zfit.Data.from_tensor(obs=obs1, tensor=z.constant(1.0, shape=(100,)))

    nll = UnbinnedNLL(model=[gauss1, gauss2], data=[data1, data2])

    def loss_funcparam1(values):
        with zfit.param.set_values(param2, values):
            return nll.value()

    def loss_funcparam1and2(values):
        with zfit.param.set_values([param2, param1], values):
            return nll.value()

    gradient_func_num = Gradient(loss_funcparam1)

    gradient1_num = gradient_func_num([param2])
    gradient1 = nll.gradient(params=param2, numgrad=numgrad)
    grad_num_truth = jacobi.jacobi(loss_funcparam1, [param2])[0]
    np.testing.assert_allclose(gradient1, grad_num_truth, rtol=1e-6)
    np.testing.assert_allclose(gradient1, gradient1_num, rtol=1e-6)
    param1.set_value(initial1)
    param2.set_value(initial2)
    params = [param2, param1]
    gradient2 = nll.gradient(params=params, numgrad=numgrad)
    gradient_func1and2 = Gradient(loss_funcparam1and2, order=2, base_step=0.1)
    gradient2_true_numdiff = gradient_func1and2([initial2, initial1])
    gradient2_true = jacobi.jacobi(loss_funcparam1and2, [initial2, initial1])[0]
    np.testing.assert_allclose(gradient2_true, gradient2_true_numdiff, rtol=1e-6)  # if this fails, numdiff/jacobi disagree
    np.testing.assert_allclose(gradient2, gradient2_true, rtol=1e-5)

    param1.set_value(initial1)
    param2.set_value(initial2)

    params3 = nll.get_params()
    gradient3 = nll.gradient(params3)
    gradients_true3 = []
    for param_o in params3:
        for param, grad in zip(params, gradient2):
            if param_o is param:
                gradients_true3.append(float(grad))
                break
    np.testing.assert_allclose(gradient3, gradients_true3, rtol=1e-6)


def test_simple_loss():
    true_a = 1.0
    true_b = 4.0
    true_c = -0.3
    truevals = true_a, true_b, true_c
    a_param = zfit.Parameter(
        "variable_a15151loss", 1.5, -1.0, 20.0, stepsize=z.constant(0.1)
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

    assert set(loss_deps.get_params()) == set(param_list)

    loss_tensor = loss_func(param_list)
    loss_value_np = loss_tensor

    assert pytest.approx(loss_value_np) == loss.value()
    assert pytest.approx(loss_value_np) == loss_deps.value()

    assert pytest.approx(
        loss_deps.value(full=True)
    ) == loss.value(full=True)

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=loss)
    assert result.valid
    assert pytest.approx(result.params[a_param]["value"], rel=0.03) == true_a
    assert pytest.approx(result.params[b_param]["value"], rel=0.06) == true_b
    assert pytest.approx(result.params[c_param]["value"], rel=0.5) == true_c

    zfit.param.set_values(param_list, znp.asarray(param_list) + 0.6)
    result2 = minimizer.minimize(loss=loss2)
    assert result2.valid
    params = list(result2.params)
    assert pytest.approx(result2.params[params[0]]["value"], rel=0.03) == true_a
    assert pytest.approx(result2.params[params[1]]["value"], rel=0.06) == true_b
    assert pytest.approx(result2.params[params[2]]["value"], rel=0.5) == true_c


def test_simple_loss_addition():

    pa = zfit.Parameter("pa", 1.0)
    pb = zfit.Parameter("pb", 2.0)
    pc = zfit.Parameter("pc", 3.0)
    pd = zfit.Parameter("pd", 4.0)
    pe = zfit.Parameter("pe", 5.0)

    loss1 = zfit.loss.SimpleLoss(lambda x: x[0] ** 2, params=pa, errordef=0.5)
    loss2 = zfit.loss.SimpleLoss(lambda x: x[0] ** 2 + x[1] ** 2, params=[pb, pc], errordef=0.5)
    loss3 = zfit.loss.SimpleLoss(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, params=[pa, pb, pc], errordef=2)
    loss4 = zfit.loss.SimpleLoss(lambda x: x[0] ** 2 + x[1] ** 2, [pd, pe], errordef=0.5)

    loss12 = loss1 + loss2

    assert set(loss12.get_params()) == {pa, pb, pc}
    assert loss12.errordef == 0.5
    assert loss12.value() == loss1.value() + loss2.value()
    assert loss12.value(full=True) == loss1.value(full=True) + loss2.value(full=True)

    loss123 = loss1 + loss2 + loss3
    assert set(loss123.get_params()) == {pa, pb, pc}
    assert loss123.errordef == 0.5
    assert loss123.value() == loss1.value() + loss2.value() + 0.25 * loss3.value()
    assert loss123.value(full=True) == loss1.value(full=True) + loss2.value(full=True) + 0.25 * loss3.value(full=True)

    loss1234 = loss1 + loss2 + loss3 + loss4
    assert set(loss1234.get_params()) == {pa, pb, pc, pd, pe}
    assert loss1234.errordef == 0.5
    assert loss1234.value() == loss1.value() + loss2.value() + 0.25 * loss3.value() + loss4.value()
    assert loss1234.value(full=True) == loss1.value(full=True) + loss2.value(full=True) + 0.25 * loss3.value(full=True) + loss4.value(full=True)

    loss3124 = loss3 + loss1 + loss2 + loss4
    assert set(loss3124.get_params()) == {pa, pb, pc, pd, pe}
    assert loss3124.errordef == 2
    assert loss3124.value() ==  loss3.value() + 4 * loss1.value() + 4 * loss2.value() + 4 * loss4.value()
    assert loss3124.value(full=True) == loss3.value(full=True) + 4 * loss1.value(full=True) + 4 * loss2.value(full=True) + 4 * loss4.value(full=True)





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
        constraints=zfit.constraint.GaussianConstraint(mu1, 1.0, sigma=0.1),
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
    x = np.array(znp.asarray(params)) + 0.1
    value_loss = loss(x)
    with zfit.param.set_values(params, x):
        true_val = zfit.run(loss.value(full=True))
        _ = zfit.run(loss.value(full=True))
        assert pytest.approx(value_loss) == true_val
        with pytest.raises(BehaviorUnderDiscussion):
            assert pytest.approx((loss())) == true_val

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
    x = znp.asarray(params) + 0.1
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
    assert pytest.approx(result.fmin.fval, abs=0.03) == result_zfit.fmin


@pytest.mark.flaky(3)
@pytest.mark.parametrize(
    "weights", [None, np.random.normal(loc=1.0, scale=0.1, size=10000)]
)
# @pytest.mark.parametrize("weights", [None])
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

    assert pytest.approx(np.mean(test_values_np), rel=rel_error) == params[mu1]["value"]
    assert pytest.approx(
        np.std(test_values_np), rel=rel_error
    ) == params[sigma1]["value"]

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


def test_loss_from_data_outside_fails():

    obs = zfit.Space("obs1", limits=(-15, 25))
    pdf = zfit.pdf.Gauss(1.0, 1.0, obs=obs)
    data = np.linspace(-20, 20, 1000)

    with pytest.raises(IntentionAmbiguousError):
        zfit.loss.UnbinnedNLL(model=pdf, data=data)



    loss = zfit.loss.UnbinnedNLL(model=pdf, data=zfit.Data(data, obs=obs.obs))
    assert loss.value() ** 2 > -1  # just a sanity check
    loss2 = zfit.loss.UnbinnedNLL(model=pdf, data=zfit.Data(data, obs=obs))
    assert loss2.value() ** 2 > -1  # just a sanity check

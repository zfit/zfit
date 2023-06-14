#  Copyright (c) 2023 zfit
import pickle
import platform
import sys

import numpy as np
import pytest

import zfit
from zfit import z
from zfit.minimizers.errors import compute_errors
from zfit.minimizers.fitresult import FitResult

true_a = 3.0
true_b = 1.1
true_c = -0.3

true_val = [true_a, true_b, true_c]


def minimizer_ids(minimizer_class_and_kwargs):
    return minimizer_class_and_kwargs[0].__name__.split(".")[-1]


def create_loss(n=15000, weights=None):
    avalue = 1.5
    a_param = zfit.Parameter(
        "variable_a15151", avalue, -1.0, 20.0, step_size=z.constant(0.1)
    )
    a_param.init_val = avalue
    bvalue = 1.5
    b_param = zfit.Parameter("variable_b15151", bvalue, 0, 20)
    b_param.init_val = bvalue
    cvalue = -0.04
    c_param = zfit.Parameter("variable_c15151", cvalue, -1, 0.0)
    c_param.init_val = cvalue
    obs1 = zfit.Space(obs="obs1", limits=(-2.4, 9.1))

    # load params for sampling
    a_param.set_value(true_a)
    b_param.set_value(true_b)
    c_param.set_value(true_c)

    gauss1 = zfit.pdf.Gauss(mu=a_param, sigma=b_param, obs=obs1)
    exp1 = zfit.pdf.Exponential(lam=c_param, obs=obs1)

    sum_pdf1 = zfit.pdf.SumPDF((gauss1, exp1), 0.2)

    sampled_data = sum_pdf1.create_sampler(n=n)
    sampled_data.resample()

    if weights is not None:
        sampled_data.set_weights(weights)

    loss = zfit.loss.UnbinnedNLL(model=sum_pdf1, data=sampled_data)

    return loss, (a_param, b_param, c_param)


def create_fitresult(minimizer_class_and_kwargs, n=15000, weights=None):
    loss, (a_param, b_param, c_param) = create_loss(n=n, weights=weights)

    true_minimum = loss.value().numpy()

    all_params = [a_param, b_param, c_param]
    for param in all_params:
        param.assign(param.init_val)  # reset the value

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)

    result = minimizer.minimize(loss=loss)
    cur_val = loss.value().numpy()
    aval, bval, cval = (result.params[p]["value"] for p in all_params)

    ret = {
        "result": result,
        "true_min": true_minimum,
        "cur_val": cur_val,
        "a": aval,
        "b": bval,
        "c": cval,
        "a_param": a_param,
        "b_param": b_param,
        "c_param": c_param,
    }

    return ret


@pytest.mark.parametrize("do_pickle", [True, False])
def test_set_values_fitresult(do_pickle):
    upper1 = 5.33
    lower1 = 0.0
    param1 = zfit.Parameter("param1", 2.0, lower1, upper1)
    param1.set_value(lower1)
    assert pytest.approx(zfit.run(param1.value())) == lower1
    param1.set_value(upper1)
    assert pytest.approx(zfit.run(param1.value())) == upper1
    with pytest.raises(ValueError):
        param1.set_value(lower1 - 0.001)
    with pytest.raises(ValueError):
        param1.set_value(upper1 + 0.001)

    fitresult = create_fitresult((zfit.minimize.Minuit, {}, True))
    result = fitresult["result"]
    param_b = fitresult["b_param"]
    param_c = fitresult["c_param"]

    val_b = fitresult["b"]
    val_c = fitresult["c"]
    with pytest.raises(ValueError):
        param_b.set_value(999)
    param_c.assign(9999)
    if do_pickle:
        result.freeze()
        result = pickle.loads(pickle.dumps(result))
    with zfit.param.set_values([param_c, param_b], values=result):
        assert zfit.run(param_b.value()) == val_b
        assert zfit.run(param_c.value()) == val_c

    # test partial
    param_new = zfit.Parameter("param_new", 42.0, 12.0, 48.0)
    with pytest.raises(ValueError):
        zfit.param.set_values(
            [param_c, param_new], values=result
        )  # allow_partial by default false

    zfit.param.set_values(
        [param_c, param_new], values=result, allow_partial=True
    )  # allow_partial by default false
    assert zfit.run(param_c.value()) == val_c

    # test partial in case we have nothing to set
    param_d = zfit.Parameter("param_d", 12)
    with pytest.raises(ValueError):
        zfit.param.set_values([param_d], result)
    zfit.param.set_values([param_d], result, allow_partial=True)


minimizers = [
    (zfit.minimize.ScipyTrustConstrV1, {}, True),
    (zfit.minimize.Minuit, {}, True),
]
if sys.version_info[1] < 11:
    minimizers.append(
        (zfit.minimize.NLoptLBFGSV1, {}, True)
    )  # TODO: nlopt for Python 3.11
    # https://github.com/DanielBok/nlopt-python/issues/19
if not platform.system() in (
    "Darwin",
    "Windows",
):  # TODO: Ipyopt installation on macosx not working
    minimizers.append((zfit.minimize.IpyoptV1, {}, False))
# sort for xdist: https://github.com/pytest-dev/pytest-xdist/issues/432
minimizers = sorted(minimizers, key=lambda val: repr(val))


@pytest.mark.parametrize(
    "minimizer_class_and_kwargs", minimizers, ids=lambda val: val[0].__name__
)
def test_freeze(minimizer_class_and_kwargs):
    result = create_fitresult(minimizer_class_and_kwargs)["result"]

    try:
        pickle.dumps(result)
    except Exception:
        pass
    result.covariance()
    result.errors()
    result.hesse()
    result.freeze()

    dumped = pickle.dumps(result)
    loaded = pickle.loads(dumped)
    test = loaded
    true = result
    assert test.fmin == true.fmin
    assert test.edm == true.edm
    assert [val for val in test.params.values()] == [
        val for val in true.params.values()
    ]
    assert test.valid == true.valid
    assert test.status == true.status
    assert test.message == true.message
    assert test.converged == true.converged
    assert test.params_at_limit == true.params_at_limit


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_fmin(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results["result"]
    assert pytest.approx(results["cur_val"]) == result.fmin


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_params(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results["result"]
    np.testing.assert_allclose(true_val, result.values, rtol=0.2)
    for param in result.params:
        assert result.values[param] == result.params[param]["value"]
        assert result.values[param.name] == result.params[param]["value"]
        assert result.values[param.name] == result.params[param.name]["value"]


def test_params_at_limit():
    loss, (param_a, param_b, param_c) = create_loss(n=5000)
    old_lower = param_a.lower
    param_a.lower = param_a.upper
    param_a.assign(param_a.upper + 5)
    minimizer = zfit.minimize.Minuit(gradient=True, tol=10.0)
    result = minimizer.minimize(loss)
    param_a.assign(100)
    assert param_a.at_limit
    assert result.params_at_limit
    param_a.lower = old_lower
    assert param_a.at_limit
    assert not result.valid


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
@pytest.mark.parametrize("use_weights", [False, True], ids=["no_weights", "weights"])
def test_covariance(minimizer_class_and_kwargs, use_weights):
    n = 15000
    if use_weights:
        weights = np.random.normal(1, 0.001, n)
    else:
        weights = None

    results = create_fitresult(
        minimizer_class_and_kwargs=minimizer_class_and_kwargs, n=n, weights=weights
    )
    result = results["result"]
    hesse = result.hesse()
    a = results["a_param"]
    b = results["b_param"]
    c = results["c_param"]

    with pytest.raises(KeyError):
        result.covariance(params=[a, b, c], method="hesse")

    cov_mat_3 = result.covariance(params=[a, b, c])
    cov_mat_2 = result.covariance(params=[c, b])
    cov_dict = result.covariance(params=[a, b, c], as_dict=True)

    assert pytest.approx(hesse[a]["error"], rel=0.01) == np.sqrt(cov_dict[(a, a)])
    assert pytest.approx(hesse[a]["error"], rel=0.01) == np.sqrt(cov_mat_3[0, 0])

    assert pytest.approx(hesse[b]["error"], rel=0.01) == np.sqrt(cov_dict[(b, b)])
    assert pytest.approx(hesse[b]["error"], rel=0.01) == np.sqrt(cov_mat_3[1, 1])
    assert pytest.approx(hesse[b]["error"], rel=0.01) == np.sqrt(cov_mat_2[1, 1])

    assert pytest.approx(hesse[c]["error"], rel=0.01) == np.sqrt(cov_dict[(c, c)])
    assert pytest.approx(hesse[c]["error"], rel=0.01) == np.sqrt(cov_mat_3[2, 2])
    assert pytest.approx(hesse[c]["error"], rel=0.01) == np.sqrt(cov_mat_2[0, 0])

    if use_weights:
        rtol, atol = 0.15, 0.015
    else:
        rtol, atol = 0.05, 0.005

    cov_mat_3_np = result.covariance(params=[a, b, c], method="hesse_np")
    np.testing.assert_allclose(cov_mat_3, cov_mat_3_np, rtol=rtol, atol=atol)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_correlation(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results["result"]
    hesse = result.hesse()
    a = results["a_param"]
    b = results["b_param"]
    c = results["c_param"]

    cor_mat = result.correlation(params=[a, b, c])
    cov_mat = result.covariance(params=[a, b, c])
    cor_dict = result.correlation(params=[a, b], as_dict=True)

    np.testing.assert_allclose(np.diag(cor_mat), 1.0)

    a_error = hesse[a]["error"]
    b_error = hesse[b]["error"]
    assert pytest.approx(cor_mat[0, 1], rel=0.01) == cov_mat[0, 1] / (a_error * b_error)
    assert pytest.approx(cor_dict[(a, b)], rel=0.01) == cov_mat[0, 1] / (
        a_error * b_error
    )


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
@pytest.mark.parametrize("cl", [None, 0.683, 0.8, 0.95, 0.9])
@pytest.mark.timeout(120)  # if stuck finding new minima
def test_errors(minimizer_class_and_kwargs, cl):
    n_max_trials = 5  # how often to try to find a new minimum
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results["result"]
    a = results["a_param"]
    b = results["b_param"]
    c = results["c_param"]

    for n_trial in range(n_max_trials):
        z_errors, new_result = result.errors(
            method="zfit_error", cl=cl, name="zfit_error"
        )
        minos_errors, _ = result.errors(
            method="minuit_minos", cl=cl, name="minuit_minos"
        )
        if new_result is None:
            break
        else:
            result = new_result
    else:  # no break occured
        assert False, "Always a new minimum was found, cannot perform test."

    for param in [a, b, c]:
        z_error_param = z_errors[param]
        minos_errors_param = minos_errors[param]
        for dir in ["lower", "upper"]:
            assert (
                pytest.approx(z_error_param[dir], rel=0.03) == minos_errors_param[dir]
            )

    with pytest.raises(KeyError):
        result.errors(method="error")


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_new_minimum(minimizer_class_and_kwargs):
    loss, params = create_loss(10000)

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)
    if isinstance(minimizer, zfit.minimize.NLoptLBFGSV1):
        return  # TODO: fix this, nlopt lbfgs cannot find the minimum when starting so close...
    a_param, b_param, c_param = params

    if test_error:
        b_param.floating = False
        b_param.set_value(3.7)
        c_param.floating = False
        result = minimizer.minimize(loss=loss)
        b_param.floating = True
        c_param.floating = True

        params_dict = {p: p.numpy() for p in params}
        hacked_result = FitResult(
            params=params_dict,
            edm=result.edm,
            fmin=result.fmin,
            info=result.info,
            loss=loss,
            status=result.status,
            converged=result.converged,
            valid=True,
            message="hacked for unittest",
            niter=999,
            criterion=None,
            minimizer=minimizer.copy(),
        )

        method = lambda **kwgs: compute_errors(covariance_method="hesse_np", **kwgs)

        errors, new_result = hacked_result.errors(
            params=params, method=method, name="interval"
        )

        assert new_result is not None

        assert hacked_result.valid is False
        for p in params:
            assert errors[p] == "Invalid, a new minimum was found."

        assert new_result.valid is True
        errors, _ = new_result.errors()
        for param in params:
            assert errors[param]["lower"] < 0
            assert errors[param]["upper"] > 0

#  Copyright (c) 2024 zfit
import pickle
import platform
import sys

import numpy as np
import pytest

import zfit
import zfit.z.numpy as znp
from zfit.minimizers.errors import compute_errors
from zfit.minimizers.fitresult import FitResult

true_a = 2.4
true_b = 1.1
true_c = -0.3

true_val = [true_a, true_b, true_c]
true_ntot = 15000
true_yields = [0.8 * true_ntot, 0.2 * true_ntot]


def minimizer_ids(minimizer_class_and_kwargs):
    return minimizer_class_and_kwargs[0].__name__.split(".")[-1]


def create_loss(n=15000, weights=None, extended=None, constraints=None):
    avalue = 1.5
    a_param = zfit.Parameter("variable_a15151", avalue, -1.0, 20.0, stepsize=0.1)
    a_param.init_val = avalue
    bvalue = 1.9
    b_param = zfit.Parameter("variable_b15151", bvalue, 0, 20)
    b_param.init_val = bvalue
    cvalue = -0.03
    c_param = zfit.Parameter("variable_c15151", cvalue, -1, 0.0)
    c_param.init_val = cvalue
    obs1 = zfit.Space(obs="obs1", limits=(-2.4, 9.1))

    # load params for sampling
    a_param.set_value(true_a)
    b_param.set_value(true_b)
    c_param.set_value(true_c)

    if extended:
        yieldgauss_init = n * 0.85
        yieldgauss = (
            zfit.Parameter("yieldgauss", yieldgauss_init, -100, n * 2)
            if extended
            else None
        )
        yieldgauss.init_val = yieldgauss_init
        yieldexp_init = n * 0.23
        yieldexp = (
            zfit.Parameter("yieldepx", yieldexp_init, -100, n) if extended else None
        )
        yieldexp.init_val = yieldexp_init
    else:
        yieldgauss = None
        yieldexp = None
    gauss1 = zfit.pdf.Gauss(mu=a_param, sigma=b_param, obs=obs1, extended=yieldgauss)
    exp1 = zfit.pdf.Exponential(lam=c_param, obs=obs1, extended=yieldexp)

    fracs = None if extended else 0.2
    sum_pdf1 = zfit.pdf.SumPDF((gauss1, exp1), fracs=fracs)

    if extended:
        yieldgauss.set_value(true_yields[0])
        yieldexp.set_value(true_yields[1])

    sampled_data = sum_pdf1.create_sampler(n=n)
    sampled_data.resample()

    if weights is not None:
        sampled_data = sampled_data.with_weights(weights)

    constraint = (
        zfit.constraint.GaussianConstraint(a_param, true_a, sigma=0.1)
        if constraints
        else None
    )
    Loss = zfit.loss.ExtendedUnbinnedNLL if extended else zfit.loss.UnbinnedNLL
    loss = Loss(model=sum_pdf1, data=sampled_data, constraints=constraint)

    if extended:
        params = (a_param, b_param, c_param, yieldgauss, yieldexp)
    else:
        params = (a_param, b_param, c_param)
    return loss, params


def create_fitresult(
    minimizer_class_and_kwargs, n=15000, weights=None, extended=None, constraints=None
):
    loss, all_params = create_loss(
        n=n, weights=weights, extended=extended, constraints=constraints
    )

    true_minimum = loss.value(full=False)

    if extended:
        a_param, b_param, c_param, yieldgauss, yieldexp = all_params
    else:
        a_param, b_param, c_param = all_params

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)

    for param in all_params:
        param.assign(param.init_val)  # reset the value
    for ntry in range(3):
        result = minimizer.minimize(loss=loss)
        cur_val = loss.value(full=False)
        if result.valid:
            break
        else:  # vary param.init_val slightly
            for param in all_params:
                param.assign(param.init_val + np.random.normal(scale=0.1))
    else:
        assert (
            False
        ), "Tried to minimize but failed 3 times, this is treated as an error."
    assert cur_val < true_minimum + 0.1, "Fit did not converge to true minimum"
    aval, bval, cval = (
        result.params[p]["value"] for p in all_params[:3]
    )  # not including yields

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
    if extended:
        ret["yieldgauss"] = yieldgauss
        ret["yieldexp"] = yieldexp
        ret["ngauss"] = result.params[yieldgauss]["value"]
        ret["nexp"] = result.params[yieldexp]["value"]

    return ret

def create_fitresult_dilled(*args, **kwargs):
    return zfit.dill.dumps(create_fitresult(*args, **kwargs))


@pytest.mark.parametrize("do_pickle", [False, "pickle", "dill"], ids=["no_pickle", "pickle", "dill"])
@pytest.mark.parametrize("weights", [None, np.random.normal(1, 0.1, true_ntot)])
@pytest.mark.parametrize("extended", [True, False], ids=["extended", "not_extended"])
@pytest.mark.parametrize("fitres_creator", [create_fitresult, create_fitresult_dilled])
def test_set_values_fitresult(do_pickle, weights, extended, fitres_creator):
    upper1 = 5.33
    lower1 = 0.0
    param1 = zfit.Parameter("param1", 2.0, lower1, upper1)
    param1.set_value(lower1)
    assert pytest.approx(znp.asarray(param1.value())) == lower1
    param1.set_value(upper1)
    assert pytest.approx(znp.asarray(param1.value())) == upper1
    with pytest.raises(ValueError):
        param1.set_value(lower1 - 0.001)
    with pytest.raises(ValueError):
        param1.set_value(upper1 + 0.001)

    fitresult = fitres_creator(
        (zfit.minimize.Minuit, {}, True), weights=weights, extended=extended
    )
    if fitres_creator is create_fitresult_dilled:
        fitresult = zfit.dill.loads(fitresult)
    result = fitresult["result"]
    param_b = fitresult["b_param"]
    param_c = fitresult["c_param"]

    val_b = fitresult["b"]
    val_c = fitresult["c"]
    with pytest.raises(ValueError):
        param_b.set_value(999)
    param_c.assign(9999)
    if do_pickle == "pickle":
        result.freeze()
        result = pickle.loads(pickle.dumps(result))
    elif do_pickle == "dill":
        zfit.dill.loads(zfit.dill.dumps(result))
    with zfit.param.set_values([param_c, param_b], values=result):
        assert znp.asarray(param_b.value()) == val_b
        assert znp.asarray(param_c.value()) == val_c

    # test partial
    param_new = zfit.Parameter("param_new", 42.0, 12.0, 48.0)
    with pytest.raises(ValueError):
        zfit.param.set_values(
            [param_c, param_new], values=result
        )  # allow_partial by default false

    zfit.param.set_values(
        [param_c, param_new], values=result, allow_partial=True
    )  # allow_partial by default false
    assert znp.asarray(param_c.value()) == val_c

    # test partial in case we have nothing to set
    param_d = zfit.Parameter("param_d", 12)
    with pytest.raises(ValueError):
        zfit.param.set_values([param_d], result)
    zfit.param.set_values([param_d], result, allow_partial=True)


minimizers = [
    (zfit.minimize.ScipyTrustConstr, {}, True),
    (zfit.minimize.Minuit, {}, True),
]


if (platf := platform.system()) not in ("Darwin",):
    # TODO: Ipyopt installation on macosx not working
    minimizers.append(
        (zfit.minimize.NLoptLBFGS, {}, True),
    )
    if platf not in ("Windows",):
        minimizers.append((zfit.minimize.Ipyopt, {}, False))
# sort for xdist: https://github.com/pytest-dev/pytest-xdist/issues/432
minimizers = sorted(minimizers, key=lambda val: repr(val))


@pytest.mark.parametrize(
    "minimizer_class_and_kwargs", minimizers, ids=lambda val: val[0].__name__
)
@pytest.mark.parametrize("dill", [False, True], ids=["no_dill", "dill"])
@pytest.mark.parametrize("weights", [np.random.normal(1, 0.1, true_ntot), None])
@pytest.mark.parametrize("extended", [True, False], ids=["extended", "not_extended"])
def test_freeze(minimizer_class_and_kwargs, dill, weights, extended):
    result = create_fitresult(
        minimizer_class_and_kwargs, weights=weights, extended=extended
    )["result"]

    if dill:
        if isinstance(result.minimizer, zfit.minimize.Ipyopt):
            with pytest.raises(zfit.exception.IpyoptPicklingError):
                _ = zfit.dill.loads(zfit.dill.dumps(result))
            pytest.skip("Ipyopt cannot be pickled")
        result = zfit.dill.loads(zfit.dill.dumps(result))

    try:
        pickle.dumps(result)
    except Exception:
        pass
    result.covariance()
    if dill:
        result = zfit.dill.loads(zfit.dill.dumps(result))
    result.errors()
    if dill:
        result = zfit.dill.loads(zfit.dill.dumps(result))
    result.hesse()
    if dill:
        result = zfit.dill.loads(zfit.dill.dumps(result))
    result.freeze()
    if dill:
        result = zfit.dill.loads(zfit.dill.dumps(result))

    dumped = pickle.dumps(result)
    loaded = pickle.loads(dumped)
    test = loaded
    true = result
    assert test.fminopt == true.fminopt
    assert test.edm == true.edm

    for testval, trueval in zip(test.params.values(), true.params.values()):
        assert testval == trueval

    assert test.valid == true.valid
    assert test.status == true.status
    assert test.message == true.message
    assert test.converged == true.converged
    assert test.params_at_limit == true.params_at_limit


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_fmin(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results["result"]
    assert pytest.approx(results["cur_val"]) == result.fminopt


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

def test_result_update_params():
    loss, (param_a, param_b, param_c) = create_loss(n=5000)
    params = [param_a, param_b, param_c]
    initial = np.array(params)
    with zfit.run.experimental_disable_param_update(True):
        minimizer = zfit.minimize.Minuit(gradient=True, tol=10.0)
        result = minimizer.minimize(loss)
        assert np.allclose(initial, np.array(params))

    result2 = result.update_params()
    assert result2 is result
    assert not np.allclose(initial, np.array(params))
    np.testing.assert_allclose(result.values, np.array(params))
    zfit.param.set_values(params, initial)
    with result:
        np.testing.assert_allclose(result.values, params)
    np.testing.assert_allclose(initial, params)






@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
@pytest.mark.parametrize("use_weights", [False, True], ids=["no_weights", "weights"])
@pytest.mark.parametrize("extended", [True, False], ids=["extended", "not_extended"])
def test_covariance(minimizer_class_and_kwargs, use_weights, extended):
    n = true_ntot
    if use_weights:
        weights = np.random.normal(1, 0.001, n)
    else:
        weights = None

    results = create_fitresult(
        minimizer_class_and_kwargs=minimizer_class_and_kwargs,
        n=n,
        weights=weights,
        extended=extended,
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


minimizers_test_errors = [
    (zfit.minimize.Minuit, {}, True),
]


@pytest.mark.parametrize(
    "minimizer_class_and_kwargs", minimizers_test_errors, ids=minimizer_ids
)
@pytest.mark.parametrize(
    "cl",
    [None, 0.683, 0.8, 0.95, 0.9],
    ids=["cldefault", "cl0683", "cl080", "cl095", "cl090"],
)
# @pytest.mark.timeout(1200)  # if stuck finding new minima
@pytest.mark.parametrize("extended", [True, False], ids=["extended", "not_extended"])
@pytest.mark.parametrize(
    "weights",
    [None, np.random.normal(1, 0.01, true_ntot)],
    ids=["no_weights", "weights"],
)
@pytest.mark.parametrize(
    "constraints", [False, True], ids=["no_constraints", "constraints"]
)
def test_errors(minimizer_class_and_kwargs, cl, weights, extended, constraints):
    n_max_trials = 5  # how often to try to find a new minimum
    results = create_fitresult(
        minimizer_class_and_kwargs=minimizer_class_and_kwargs,
        weights=weights,
        extended=extended,
        constraints=constraints,
    )
    result = results["result"]
    a = results["a_param"]
    b = results["b_param"]
    c = results["c_param"]
    params = list(result.params.keys())
    relerr = 0.03 if weights is None else 0.05

    for n_trial in range(n_max_trials):
        with zfit.run.set_autograd_mode(False):
            z_errors, new_result = result.errors(
                method="zfit_error",
                cl=cl,
                name="zfit_error",
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

    for param in params:
        z_error_param = z_errors[param]
        minos_errors_param = minos_errors[param]
        if weights is not None and extended:
            continue  # TODO: fix this, somehow the uncertainty is not good for this case
        for dir in ["lower", "upper"]:
            assert (
                pytest.approx(z_error_param[dir], rel=relerr) == minos_errors_param[dir]
            )

    with pytest.raises(KeyError):
        result.errors(method="error")


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers, ids=minimizer_ids)
def test_new_minimum(minimizer_class_and_kwargs):
    loss, params = create_loss(10000)

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)
    if isinstance(minimizer, zfit.minimize.NLoptLBFGS):
        return  # TODO: fix this, nlopt lbfgs cannot find the minimum when starting so close...
    a_param, b_param, c_param = params

    if test_error:
        b_param.floating = False
        b_param.set_value(3.7)
        c_param.floating = False
        result = minimizer.minimize(loss=loss)
        b_param.floating = True
        c_param.floating = True

        params_dict = {p: float(p) for p in params}
        hacked_result = FitResult(
            params=params_dict,
            edm=result.edm,
            fminopt=result.fminopt,
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

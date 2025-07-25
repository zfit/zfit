#  Copyright (c) 2025 zfit
import itertools
import platform
import sys

import numpy as np
import pytest
import scipy.optimize
from ordered_set import OrderedSet

import zfit.minimizers.optimizers_tf
import zfit.z.numpy as znp
from zfit.minimizers.base_tf import WrapOptimizer
from zfit.util.exception import OperationNotAllowedError

true_mu = 4.5
true_sigma = 2
true_lambda: float = -0.03

parameter_tol = 0.1
max_distance_to_min = 0.5


def create_loss(obs1):
    mu_param = zfit.Parameter("mu", true_mu + 0.85, -15.0, 15, stepsize=0.03)
    sigma_param = zfit.Parameter("sigma", true_sigma * 0.62, 0.01, 50, stepsize=0.03)
    lambda_param = zfit.Parameter(
        "lambda", true_lambda * 0.69, -0.51, -0.0003, stepsize=0.001
    )

    gauss1 = zfit.pdf.Gauss(mu=mu_param, sigma=sigma_param, obs=obs1)
    exp1 = zfit.pdf.Exponential(lam=lambda_param, obs=obs1)

    sum_pdf1 = zfit.pdf.SumPDF([gauss1, exp1], 0.8)
    # load params for sampling
    with mu_param.set_value(true_mu):
        with sigma_param.set_value(true_sigma):
            with lambda_param.set_value(true_lambda):
                sampled_data = sum_pdf1.create_sampler(n=25000)
                sampled_data.resample()

                loss = zfit.loss.UnbinnedNLL(
                    model=sum_pdf1, data=sampled_data, options={"subtr_const": True}
                )
                minimum = loss.value(full=False)

    return loss, minimum, (mu_param, sigma_param, lambda_param)


verbosity = None


def make_min_grad_hesse():
    minimizers = [
        zfit.minimize.ScipyTruncNC,
        zfit.minimize.ScipyTrustNCG,  # Too bad
        zfit.minimize.ScipyTrustKrylov,  # Too bad
        zfit.minimize.ScipySLSQP,
        zfit.minimize.ScipyLBFGSB,
        zfit.minimize.ScipyTrustConstr,
    ]
    min_options = []
    for opt in minimizers:
        grad = opt._VALID_SCIPY_GRADIENT
        hess = opt._VALID_SCIPY_HESSIAN
        if not grad:
            grad = {None}  # the default is None, so this will skip it
        if not hess:
            hess = {None}
        product = itertools.product([opt], grad, hess)
        min_options.extend(product)
    return min_options


@pytest.mark.parametrize("minimizer_gradient_hessian", make_min_grad_hesse(), ids=lambda x: f"{x[0].__name__}_{x[1]}_{x[2]}")
@pytest.mark.flaky(reruns=3)
def test_scipy_derivative_options(minimizer_gradient_hessian):
    minimizer_cls, gradient, hessian = minimizer_gradient_hessian
    def loss(params):
        return params[0] ** 2. + params[1] ** 2.

    loss.errordef = 0.5
    true_min = 0.
    kwargs = {}

    if gradient is not None:
        kwargs["gradient"] = gradient
    if hessian is not None:
        kwargs["hessian"] = hessian

    try:
        minimizer = minimizer_cls(**kwargs)
    except ValueError as error:  # we test a not allowed combination
        if "Whenever the gradient is estimated via finite-differences" in error.args[0]:
            return
        else:
            raise

    result = minimizer.minimize(loss=loss, params={"a": 0.5, "b": 0.2})
    assert result.valid

    found_min = loss(result.x)
    assert true_min + max_distance_to_min >= found_min

    np.testing.assert_allclose([0., 0.], result.x, atol=0.02)
    assert result.converged


do_errors_most = False


def minimizer_ids(minimizer_class_and_kwargs):
    return minimizer_class_and_kwargs[0].__name__.split(".")[-1]


minimizers = [
    # minimizers, minimizer_kwargs, do error estimation
    # TensorFlow minimizers
    # (zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)),
    #  False),
    # (
    #     zfit.minimize.Adam,
    #     dict(learning_rate=0.05, verbosity=verbosity, tol=0.00001),
    #     False,
    # ),  # Not supported anymore, needs Keras 3 interface update
    # Minuit minimizer
    (
        zfit.minimize.Minuit,
        {"verbosity": verbosity},
        {"error": True, "longtests": True},
    ),  # works
    # Ipyopt minimizer
    # TensorFlow Probability minimizer
    # (BFGS, {}, True),  # doesn't work as it uses the graph, violates assumption in minimizer
    # SciPy Minimizer
    (  # TODO: reactivate. Not working, completely overshooting estimates. Maybe normalize variables?
    zfit.minimize.ScipyLBFGSB,
    {"verbosity": verbosity},
    {"error": True, "numgrad": False, "approx": True},
    ),
    (zfit.minimize.ScipyBFGS, {"verbosity": verbosity}, {"error": do_errors_most}),
    (zfit.minimize.ScipyTrustNCG, {'tol': 1e-5, "verbosity": verbosity}, True),
    (zfit.minimize.ScipyTrustKrylov, {"verbosity": verbosity}, True),  # Too unstable
    (
        zfit.minimize.ScipyTrustConstr,
        {
            "verbosity": verbosity,
        },
        {"error": True, "longtests": bool(zfit.run.get_graph_mode())},
    ),
    (
        zfit.minimize.ScipyPowell,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.ScipySLSQP,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (zfit.minimize.ScipyCOBYLA, {"verbosity": verbosity, }, {'error': do_errors_most}),  # Too bad
    (zfit.minimize.ScipyDogleg, {'tol': 1e-5, "verbosity": verbosity}, do_errors_most),  # works badly
    (zfit.minimize.ScipyNewtonCG, {"verbosity":verbosity, }, {'error': do_errors_most}),  # Too sensitive? Fails in line-search?
    (
        zfit.minimize.ScipyTruncNC,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
]
# NLopt minimizer
nlopt_minimizers = [
    (
        zfit.minimize.NLoptLBFGS,
        {
            "verbosity": verbosity,
        },
        {"error": True, "longtests": bool(zfit.run.get_graph_mode())},
    ),
    (
        zfit.minimize.NLoptTruncNewton,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptSLSQP,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptMMA,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptCCSAQ,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptSubplex,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptCOBYLA,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptMLSL,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptStoGO,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptBOBYQA,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptISRES,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptESCH,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
    (
        zfit.minimize.NLoptShiftVar,
        {
            "verbosity": verbosity,
        },
        {"error": do_errors_most},
    ),
]
zfit_minimizers = [
    (
        zfit.minimize.LevenbergMarquardt,
        {
            "verbosity": verbosity,
        },
        {"error": False, "longtests": bool(zfit.run.get_graph_mode())},
    ),]

if platform.system() not in ("Darwin",):
    minimizers.extend(nlopt_minimizers)




minimizers_small = [
    (zfit.minimize.ScipyTrustConstr, {}, True),
    (zfit.minimize.Minuit, {}, True),
    (zfit.minimize.LevenbergMarquardt, {}, True),
]
if sys.version_info[1] < 12 and platform.system() not in ("Darwin",):
    minimizers_small.append((zfit.minimize.NLoptLBFGS, {}, True))
if (
        platform.system()
        not in (
        "Darwin",
        "Windows",
)
        and sys.version_info[1] < 12
):  # TODO: Ipyopt installation on macosx not working
    # TODO: ipyopt fails? Why
    minimizers_small.append((zfit.minimize.Ipyopt, {}, False))
    minimizers.append(
        (
            zfit.minimize.Ipyopt,
            {"verbosity": verbosity},
            {"error": True, "longtests": True},
        )
    )
# To run individual minimizers
# minimizers = [(zfit.minimize.Minuit, {"verbosity": verbosity, 'gradient': True}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.IpyoptV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyLBFGSBV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyBFGS, {"verbosity": 10}, True)]
# minimizers = [(zfit.minimize.ScipyPowellV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipySLSQPV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyNelderMeadV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyCOBYLAV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyNewtonCGV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustNCGV1, {'tol': 1e-3, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTruncNCV1, {'tol': 1e-5, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyDoglegV1, {'tol': 1e3, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustConstrV1, {'tol': 1e-5, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustKrylovV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptLBFGSV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptTruncNewtonV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptSLSQPV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptMMAV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptCCSAQV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptMLSLV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptStoGOV1, {'verbosity': 7}, {'error': True, 'longtests': True})]  # DOESN'T WORK!
# minimizers = [(zfit.minimize.NLoptSubplexV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptESCHV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptISRESV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptBOBYQAV1, {'verbosity': 7}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.NLoptShiftVarV1, {'verbosity': 7, 'rank': 2}, {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.Minuit, {'verbosity': 6}, True)]
# minimizers = [(zfit.minimize.BFGS, {'verbosity': 6}, True)]
minimizers = [(zfit.minimize.LevenbergMarquardt, {'verbosity': 6}, True)]


# sort for xdist: https://github.com/pytest-dev/pytest-xdist/issues/432
minimizers = sorted(minimizers, key=lambda val: repr(val))
minimizers_small = sorted(minimizers_small, key=lambda val: repr(val))

obs1 = zfit.Space(obs="obs1", limits=(-2.4, 9.1))
obs1_split = (
        zfit.Space(obs="obs1", limits=(-2.4, 1.3))
        + zfit.Space(obs="obs1", limits=(1.3, 2.1))
        + zfit.Space(obs="obs1", limits=(2.1, 9.1))
)


def test_floating_flag():
    obs = zfit.Space("x", limits=(-2, 3))
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)
    sigma.floating = False
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    normal_np = np.random.normal(loc=2.0, scale=3.0, size=10000)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll, params=[mu, sigma])
    assert list(result.params.keys()) == [mu]
    assert sigma not in result.params


@pytest.mark.parametrize(
    "params",
    [
        # np.random.normal(size=5),
        [1.4, 0.6, 1.5],
        {
            "value": [1.4, 0.6, 1.5],
            "lower": np.ones(3) * (-5),
            "upper": np.ones(3) * (9),
            "stepsize": np.linspace(0.1, 0.2, 3),
        },
    ],
)
@pytest.mark.parametrize(
    "minimizer_class_and_kwargs",
    minimizers_small,
    ids=minimizer_ids,
)
@pytest.mark.flaky(reruns=1)
def test_minimize_pure_func(params, minimizer_class_and_kwargs):
    minimizer_class, minimizer_kwargs, _ = minimizer_class_and_kwargs
    with zfit.run.set_autograd_mode(False), zfit.run.set_graph_mode(False):
        minimizer = minimizer_class(**minimizer_kwargs)
        func = scipy.optimize.rosen
        func.errordef = 0.5
        result = minimizer.minimize(func, params)
        result.hesse(method="hesse_np", name="hesse")
        param = list(result.params)[1]
        result.errors(param, name="errors")
    assert result.valid
    assert pytest.approx(result.fmin, abs=0.01) == 0

    assert pytest.approx(result.params[param]["errors"]["lower"], rel=0.15) == -0.6
    assert pytest.approx(result.params[param]["errors"]["upper"], rel=0.15) == 0.65
    for param, error in zip(result.params, [0.32, 0.64, 1.3]):
        assert pytest.approx(result.params[param]["hesse"]["error"], rel=0.15) == error


def test_dependent_param_extraction():
    obs = zfit.Space("x", limits=(-2, 3))
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)
    sigma1 = zfit.ComposedParameter("sigma1", lambda sigma, mu: sigma + mu, params=[sigma, mu])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
    normal_np = np.random.normal(loc=2.0, scale=3.0, size=10)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    params_checked = OrderedSet(
        minimizer._check_convert_input(nll, params=[mu, sigma1])[1]
    )
    assert {mu, sigma} == set(params_checked)
    sigma.floating = False
    params_checked = minimizer._check_convert_input(nll, params=[mu, sigma1])[1]
    assert {
               mu,
           } == set(params_checked)


# @pytest.mark.run(order=4)
# chunksizes = [100000, 3000]
chunksizes = [100000]
numgrads = [False, True]
# num_grads = [True]
# num_grads = [False]

spaces_all = [obs1, obs1_split] if not zfit.run.executing_eagerly() else [obs1]

error_scales = {None: 1, 1: 1, 2: 2}


@pytest.mark.parametrize("chunksize", chunksizes)
@pytest.mark.parametrize(
    "numgrad", numgrads, ids=lambda x: "numgrad" if x else "autograd"
)
@pytest.mark.parametrize("spaces", spaces_all)
@pytest.mark.parametrize(
    "minimizer_class_and_kwargs",
    minimizers,
    ids=lambda minimizer_class_and_kwargs: minimizer_class_and_kwargs[0].__name__.split(
        "."
    )[-1],
)
@pytest.mark.flaky(reruns=1)
@pytest.mark.timeout(380)
def test_minimizers(minimizer_class_and_kwargs, chunksize, numgrad, spaces, request):
    long_clarg = request.config.getoption("--longtests")
    # long_clarg = True
    # zfit.run.chunking.active = True
    # zfit.run.chunking.max_n_points = chunksize
    with zfit.run.set_autograd_mode(not numgrad):
        minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs

        if not isinstance(test_error, dict):
            test_error = {"error": test_error}

        # numgrad = test_error.get("numgrad", False)
        do_long = test_error.get("longtests", False)
        has_approx = test_error.get("approx", False)
        test_error = test_error["error"]

        skip_tests = (
                not long_clarg
                and not do_long
                and not (
                chunksize == chunksizes[0]
                and numgrad is False
                and spaces is spaces_all[0]
        )
        )

        if skip_tests:
            return
        if not long_clarg and not do_long:
            test_error = False

        # start actual test
        obs = spaces
        loss, true_min, params = create_loss(obs1=obs)
        (mu_param, sigma_param, lambda_param) = params
        minimizer_hightol = minimizer_class(**{**minimizer_kwargs, "tol": 5.0})

        minimizer = minimizer_class(**minimizer_kwargs)

        # run 3 times: once fully (as normal, once with a high tol first, then with a low restarting from the previous one)
        init_vals = znp.asarray(params)

        result = minimizer.minimize(loss=loss)
        zfit.param.set_values(params, init_vals)
        result_hightol = minimizer_hightol.minimize(loss=loss)
        zfit.param.set_values(params, init_vals)
        result_lowtol = minimizer.minimize(loss=result_hightol)

        assert result.valid
        assert result_hightol.valid
        assert result_lowtol.valid
        found_min = loss.value(full=False)
        assert true_min + max_distance_to_min >= found_min

        assert pytest.approx(result.fminopt, abs=2.0) == result_lowtol.fminopt
        if not isinstance(minimizer, zfit.minimize.Ipyopt):
            assert (
                    result_lowtol.info["n_eval"]
                    < 1.2 * result.info["n_eval"] + 10  # +10 if it's very small, it's hard
            )  # should not be more, surely not a lot

        aval, bval, cval = (znp.asarray(v) for v in (mu_param, sigma_param, lambda_param))

        assert pytest.approx(aval, abs=parameter_tol) == true_mu
        assert pytest.approx(bval, abs=parameter_tol) == true_sigma
        assert pytest.approx(cval, abs=parameter_tol) == true_lambda
        assert result.converged

        # Test Hesse
        if test_error:
            for cl, errscale in [(0.683, 1), (0.9548, 2), (0.99747, 3)]:
                hesse_methods = ["hesse_np"]
                profile_methods = ["zfit_error"]
                from zfit.minimizers.minimizer_minuit import Minuit

                hesse_methods.append("minuit_hesse")
                profile_methods.append("minuit_minos")
                # the following minimizers should support the "approx" option as the give access to the approx Hessian
                if isinstance(
                        minimizer,
                        (
                                Minuit,
                                zfit.minimize.ScipyLBFGSB,
                                zfit.minimize.ScipyNewtonCG,
                                zfit.minimize.ScipyTruncNC,
                        ),
                ):
                    hesse_methods.append("approx")

                rel_error_tol = 0.15
                for method in hesse_methods:
                    name = f"{method}_{cl:.3g}"
                    sigma_hesse = result.hesse(
                        params=sigma_param, method=method, name=name, cl=cl
                    )
                    assert tuple(sigma_hesse.keys()) == (sigma_param,)
                    errors = result.hesse(method=method, name=name, cl=cl)
                    sigma_hesse = sigma_hesse[sigma_param]
                    can_be_none = method == "approx" and not has_approx
                    # skip if it can be None and it is None
                    sigma_error_true = 0.015 * errscale
                    if not (can_be_none and errors[sigma_param].get("error") is None):
                        assert pytest.approx(
                            sigma_error_true, abs=rel_error_tol
                        ) == abs(errors[sigma_param]["error"])
                    if not (can_be_none and errors[lambda_param].get("error") is None):
                        assert pytest.approx(
                            0.01 * errscale, abs=0.01
                        ) == abs(errors[lambda_param]["error"])
                    if not (can_be_none and sigma_hesse.get("error") is None):
                        assert pytest.approx(
                            sigma_error_true, abs=rel_error_tol
                        ) == abs(sigma_hesse["error"])

                for profile_method in profile_methods:
                    # Test Error
                    pname = f"{profile_method}_{cl:.3g}"

                    a_errors, _ = result.errors(
                        params=mu_param, method=profile_method, name=pname, cl=cl
                    )
                    assert tuple(a_errors.keys()) == (mu_param,)
                    errors, _ = result.errors(method=profile_method, name=pname, cl=cl)
                    a_error = a_errors[mu_param]
                    assert pytest.approx(-a_error["upper"], rel=0.1) == a_error["lower"]
                    assert pytest.approx(
                        -0.021 * errscale, rel=rel_error_tol
                    ) == a_error["lower"]
                    assert pytest.approx(
                        -sigma_error_true, rel=rel_error_tol
                    ) == errors[sigma_param]["lower"]
                    assert pytest.approx(
                        -0.007 * errscale, rel=rel_error_tol
                    ) == errors[lambda_param]["lower"]
                    assert pytest.approx(
                        0.007 * errscale, rel=rel_error_tol
                    ) == errors[lambda_param]["upper"]

                    assert pytest.approx(
                        a_error["lower"], rel=rel_error_tol
                    ) == errors[mu_param]["lower"]
                    assert pytest.approx(
                        a_error["upper"], rel=rel_error_tol
                    ) == errors[mu_param]["upper"]

                    # Test Error method name
                    a_errors, _ = result.errors(
                        params=mu_param, method=profile_method, name="error1"
                    )
                    assert tuple(a_errors.keys()) == (mu_param,)
                    errors, _ = result.errors(name="error42", method=profile_method)
                    a_error = a_errors[mu_param]

                    assert pytest.approx(
                        result.params[mu_param]["error42"]["lower"], rel=rel_error_tol
                    ) == a_error["lower"]
                    assert pytest.approx(
                        result.params[mu_param]["error1"]["lower"], rel=rel_error_tol
                    ) == a_error["lower"]
                    for param, errors2 in result.params.items():
                        assert pytest.approx(
                            errors2["error42"]["lower"], rel=rel_error_tol
                        ) == errors[param]["lower"]
                        assert pytest.approx(
                            errors2["error42"]["upper"], rel=rel_error_tol
                        ) == errors[param]["upper"]

                    # test custom error
                    def custom_error_func(result, params, cl):
                        return (
                            {param: {"myval": 42} for param in params},
                            None,
                        )

                    custom_errors, _ = result.errors(
                        method=custom_error_func, name="custom_method1"
                    )
                    for param, errors2 in result.params.items():
                        assert custom_errors[param]["myval"] == 42

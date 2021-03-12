#  Copyright (c) 2021 zfit
from collections import OrderedDict

import numpy as np
import pytest
import scipy.optimize
from ordered_set import OrderedSet

import zfit.minimizers.optimizers_tf
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.minimizers.base_tf import WrapOptimizer
from zfit.util.exception import OperationNotAllowedError


def teardown_function():
    zfit.settings.options['numerical_grad'] = False
    from zfit.core.testing import teardown_function as td_func
    td_func()


from zfit.minimizers.minimizer_tfp import BFGS

true_mu = 4.5
true_sigma = 2
true_lambda = -0.03


def create_loss(obs1):
    mu_param = zfit.Parameter("mu", true_mu - 2., -5., 9.,
                              step_size=0.03
                              )
    sigma_param = zfit.Parameter("sigma", true_sigma * 0.3, 0.01, 10,
                                 step_size=0.03)
    lambda_param = zfit.Parameter("lambda", true_lambda * 0.5, -0.5, -0.0003,
                                  step_size=0.001)

    gauss1 = zfit.pdf.Gauss(mu=mu_param, sigma=sigma_param, obs=obs1)
    exp1 = zfit.pdf.Exponential(lam=lambda_param, obs=obs1)

    sum_pdf1 = zfit.pdf.SumPDF([gauss1, exp1], 0.8)
    # load params for sampling
    with mu_param.set_value(true_mu):
        with sigma_param.set_value(true_sigma):
            with lambda_param.set_value(true_lambda):
                sampled_data = sum_pdf1.create_sampler(n=25000)
                # sampled_data = sum_pdf1.create_sampler(n=10000000)
                sampled_data.resample()

                loss = zfit.loss.UnbinnedNLL(model=sum_pdf1, data=sampled_data)
                minimum = loss.value().numpy()

    return loss, minimum, (mu_param, sigma_param, lambda_param)


verbosity = 7
zfit.settings.set_verbosity(verbosity=verbosity)

minimizers = [  # minimizers, minimizer_kwargs, do error estimation
    # (zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)),
    #  False),
    (zfit.minimize.Adam, dict(learning_rate=0.05, verbosity=verbosity, tol=0.00001), False),
    # works
    (zfit.minimize.Minuit, {"tol": 0.0001, "verbosity": verbosity}, {'error': True, 'longtests': True}),  # works
    # (BFGS, {}, True),  # doesn't work as it uses the graph, violates assumption in minimizer
    (zfit.minimize.ScipyLBFGSBV1, {'tol': 1e-5, "verbosity": verbosity},
     {'error': True, 'numgrad': False, 'approx': True}),
    (zfit.minimize.ScipyTrustNCGV1, {'tol': 1e-5, "verbosity": verbosity}, True),
    (zfit.minimize.ScipyDoglegV1, {'tol': 1e-5, "verbosity": verbosity}, True),
    (zfit.minimize.ScipyTrustKrylovV1, {"verbosity": verbosity}, True),
    (zfit.minimize.ScipyTrustConstrV1, {"verbosity": verbosity, }, {'error': True, 'longtests': True}),
    (zfit.minimize.ScipyPowellV1, {"verbosity": verbosity, }, {'error': True}),
    (zfit.minimize.ScipySLSQPV1, {"verbosity": verbosity, }, {'error': True}),
    # (zfit.minimize.ScipyNewtonCGV1, {"verbosity":verbosity,}, {'error': True}),  # Too sensitive? Fails in line-search?
    (zfit.minimize.ScipyTruncNCV1, {"verbosity": verbosity, }, {'error': True}),

    (zfit.minimize.NLoptLBFGSV1, {"verbosity": verbosity, }, {'error': True, 'longtests': True}),
    (zfit.minimize.NLoptTruncNewtonV1, {"verbosity": verbosity, }, True),
    (zfit.minimize.NLoptSLSQPV1, {"verbosity": verbosity, }, True),
    (zfit.minimize.NLoptMMAV1, {"verbosity": verbosity, }, True),
    (zfit.minimize.NLoptCCSAQV1, {"verbosity": verbosity, }, True),
    (zfit.minimize.NLoptSubplexV1, {"verbosity": verbosity, }, True),

    # (zfit.minimize.Scipy, {'tol': 1e-8, 'algorithm': 'L-BFGS-B'}, False),  # works not, L-BFGS_B
    # (zfit.minimize.Scipy, {'tol': 1e-8, 'algorithm': 'CG'}, False),
    # (zfit.minimize.Scipy, {'tol': 1e-8, 'algorithm': 'Powell'}, False),
    # works
    # (zfit.minimize.Scipy, {'tol': 1e-8, 'algorithm': 'BFGS'}, False),  # too bad
    # (zfit.minimize.Scipy, {'tol': 0.00001, 'algorithm': 'Newton-CG', "scipy_grad": False}, False),  # too bad
    # (zfit.minimize.Scipy, {'tol': 0.00001, 'algorithm': 'TNC'}, False),  # unstable
    # (zfit.minimize.Scipy, {'tol': 1e-8, 'algorithm': 'trust-constr'}, False),  # works
    # (zfit.minimize.Scipy, {'tol': 0.00001, 'algorithm': 'trust-ncg', "scipy_grad": True}, False),  # need Hess
    # (zfit.minimize.Scipy, {'tol': 0.00001, 'algorithm': 'trust-krylov', "scipy_grad": True}, False),  # Hess
    # (zfit.minimize.Scipy, {'tol': 0.00001, 'algorithm': 'dogleg', "scipy_grad": True}, False),  # Hess
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LD_LBFGS}, True),  # works not, why not?
    # (zfit.minimize.NLopt, {'algorithm': nlopt.GD_STOGO}, True),  # takes too long
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LN_NELDERMEAD}, True),  # performs too bad
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LN_SBPLX},True),  # works
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LD_MMA}, True),  # doesn't minimize
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LD_SLSQP}, True),  # doesn't minimize
    # (zfit.minimize.NLopt, {'tol': 1e-8, 'algorithm': nlopt.LD_TNEWTON_PRECOND_RESTART}, True),  # no minimize
    # (zfit.minimize.NLopt, {'tol': 0.0001, 'algorithm': nlopt.LD_VAR2}, True),  # doesn't minimize
]

# minimizers = [(zfit.minimize.Minuit, {"tol": 0.0001, "verbosity": verbosity, 'use_minuit_grad':False},
#                {'error': True, 'longtests': True})]
# minimizers = [(zfit.minimize.IPoptV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyLBFGSBV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyPowellV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipySLSQPV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyNelderMeadV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyNewtonCGV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustNCGV1, {'tol': 1e-3, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTruncNCV1, {'tol': 1e-5, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyDoglegV1, {'tol': 1e-5, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustConstrV1, {'tol': 1e-5, 'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.ScipyTrustKrylovV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptLBFGSV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptTruncNewtonV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptSLSQPV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptMMAV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptCCSAQV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.NLoptMLSLV1, {'verbosity': 7}, True)]  # DOESN'T WORK!
# minimizers = [(zfit.minimize.NLoptSubplexV1, {'verbosity': 7}, True)]
# minimizers = [(zfit.minimize.Minuit, {'verbosity': 6}, True)]

obs1 = zfit.Space(obs='obs1', limits=(-2.4, 9.1))
obs1_split = (zfit.Space(obs='obs1', limits=(-2.4, 1.3))
              + zfit.Space(obs='obs1', limits=(1.3, 2.1))
              + zfit.Space(obs='obs1', limits=(2.1, 9.1)))


def test_floating_flag():
    obs = zfit.Space("x", limits=(-2, 3))
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)
    sigma.floating = False
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    normal_np = np.random.normal(loc=2., scale=3., size=10000)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll, params=[mu, sigma])
    assert list(result.params.keys()) == [mu]
    assert sigma not in result.params


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers)
def test_minimize_pure_func(minimizer_class_and_kwargs):
    zfit.run.set_autograd_mode(False)
    zfit.run.set_graph_mode(False)
    minimizer_class, minimizer_kwargs, _ = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)
    func = scipy.optimize.rosen
    params = np.random.normal(size=5)
    if isinstance(minimizer, WrapOptimizer):
        with pytest.raises(OperationNotAllowedError):
            _ = minimizer.minimize(func, params)
    else:
        result = minimizer.minimize(func, params)
        assert result.valid
    # result.hesse()
    # result.errors()


def test_dependent_param_extraction():
    obs = zfit.Space("x", limits=(-2, 3))
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)
    sigma1 = zfit.ComposedParameter('sigma1', lambda sigma, mu: sigma + mu,
                                    [sigma, mu])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
    normal_np = np.random.normal(loc=2., scale=3., size=10)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    params_checked = OrderedSet(minimizer._check_convert_input(nll, params=[mu, sigma1])[1])
    assert {mu, sigma} == set(params_checked)
    sigma.floating = False
    params_checked = minimizer._check_convert_input(nll, params=[mu, sigma1])[1]
    assert {mu, } == set(params_checked)


# @pytest.mark.run(order=4)
# chunksizes = [100000, 3000]
chunksizes = [100000]
# skip the numerical gradient due to memory leak bug, TF2.3 fix: https://github.com/tensorflow/tensorflow/issues/35010
# num_grads = [bo for bo in [False, True] if not bo or zfit.run.get_graph_mode()]
num_grads = [False, True]
# num_grads = [True]
# num_grads = [False]

spaces_all = [obs1, obs1_split]


@pytest.mark.parametrize("chunksize", chunksizes)
@pytest.mark.parametrize("num_grad", num_grads)
@pytest.mark.parametrize("spaces", spaces_all)
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers)
@pytest.mark.flaky(reruns=3)
def test_minimizers(minimizer_class_and_kwargs, num_grad, chunksize, spaces,
                    pytestconfig):
    long_clarg = pytestconfig.getoption("longtests")
    # long_clarg = True
    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize
    zfit.settings.options['numerical_grad'] = num_grad

    # minimize_func(minimizer_class_and_kwargs, obs=spaces)


    parameter_tol = 0.1
    max_distance_to_min = 2.5

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs

    if not isinstance(test_error, dict):
        test_error = {'error': test_error}

    numgrad = test_error.get('numgrad', True)
    do_long = test_error.get('longtests', False)
    has_approx = test_error.get('approx', False)
    test_error = test_error['error']
    if not numgrad:
        return

    skip_tests = (not long_clarg
                  and not do_long
                  and not (chunksize == chunksizes[0]
                           and num_grad == num_grads[0]
                           and spaces is spaces_all[0]))
    if skip_tests:
        return
    if not long_clarg and not do_long:
        test_error = False

    obs = spaces
    loss, true_min, params = create_loss(obs1=obs)
    (mu_param, sigma_param, lambda_param) = params
    minimizer_hightol = minimizer_class(**{**minimizer_kwargs,
                                           'tol': 100 * minimizer_kwargs.get(
                                               'tol', 0.01)})

    minimizer = minimizer_class(**minimizer_kwargs)

    # Currently not working, stop the test here. Memory leak?
    if isinstance(minimizer, BFGS) and num_grad and not zfit.run.mode['graph']:
        return
    init_vals = zfit.run(params)
    result = minimizer.minimize(loss=loss)
    zfit.param.set_values(params, init_vals)
    result_lowtol = minimizer_hightol.minimize(loss=loss)
    zfit.param.set_values(params, init_vals)
    result_lowtol2 = minimizer.minimize(loss=result_lowtol)

    assert result.valid
    assert result_lowtol.valid
    assert result_lowtol2.valid
    found_min = loss.value().numpy()
    assert true_min + max_distance_to_min >= found_min

    assert result_lowtol2.fmin == pytest.approx(result.fmin, abs=2.)
    if not isinstance(minimizer, zfit.minimize.IpyoptV1):
        assert result_lowtol2.info['n_eval'] < 0.99 * result.info['n_eval']

    aval, bval, cval = [zfit.run(v) for v in
                        (mu_param, sigma_param, lambda_param)]

    assert true_mu == pytest.approx(aval, abs=parameter_tol)
    assert true_sigma == pytest.approx(bval, abs=parameter_tol)
    assert true_lambda == pytest.approx(cval, abs=parameter_tol)
    assert result.converged

    # Test Hesse
    if test_error:
        hesse_methods = ['hesse_np', 'approx']
        profile_methods = ['zfit_error']
        if isinstance(minimizer, zfit.minimize.Minuit):
            hesse_methods.append('minuit_hesse')
            profile_methods.append('minuit_minos')
        else:
            with pytest.raises(TypeError):
                _ = result.errors(params=mu_param, method="minuit_minos")

        rel_error_tol = 0.15
        for method in hesse_methods:
            sigma_hesse = result.hesse(params=sigma_param, method=method)
            assert tuple(sigma_hesse.keys()) == (sigma_param,)
            errors = result.hesse()
            sigma_hesse = sigma_hesse[sigma_param]
            can_be_none = method == "approx" and not has_approx
            # skip if it can be None and it is None
            sigma_error_true = 0.015
            if not (can_be_none and errors[sigma_param].get('error') is None):
                assert abs(errors[sigma_param]['error']) == pytest.approx(sigma_error_true,
                                                                          abs=rel_error_tol)
            if not (can_be_none and errors[lambda_param].get('error') is None):
                assert abs(errors[lambda_param]['error']) == pytest.approx(0.01,
                                                                           abs=0.01)
            if not (can_be_none and sigma_hesse.get('error') is None):
                assert abs(sigma_hesse['error']) == pytest.approx(sigma_error_true, abs=rel_error_tol)

        for profile_method in profile_methods:
            # Test Error
            a_errors, _ = result.errors(params=mu_param, method=profile_method)
            assert tuple(a_errors.keys()) == (mu_param,)
            errors, _ = result.errors(method=profile_method)
            a_error = a_errors[mu_param]
            assert a_error["lower"] == pytest.approx(-a_error['upper'],
                                                     rel=rel_error_tol)
            assert a_error["lower"] == pytest.approx(-0.021, rel=rel_error_tol)
            assert errors[sigma_param]["lower"] == pytest.approx(-sigma_error_true,
                                                                      rel=rel_error_tol)
            assert abs(errors[lambda_param]['lower']) == pytest.approx(0.007,
                                                                       rel=rel_error_tol)
            assert abs(errors[lambda_param]['upper']) == pytest.approx(0.007,
                                                                       rel=rel_error_tol)

            assert errors[mu_param]['lower'] == pytest.approx(a_error['lower'],
                                                              rel=rel_error_tol)
            assert errors[mu_param]['upper'] == pytest.approx(a_error['upper'],
                                                              rel=rel_error_tol)

            # Test Error method name
            a_errors, _ = result.errors(params=mu_param, method=profile_method,
                                        name='error1')
            assert tuple(a_errors.keys()) == (mu_param,)
            errors, _ = result.errors(name='error42',
                                      method=profile_method)
            a_error = a_errors[mu_param]

            assert a_error['lower'] == pytest.approx(
                result.params[mu_param]['error42']['lower'], rel=rel_error_tol)
            assert a_error['lower'] == pytest.approx(
                result.params[mu_param]['error1']['lower'], rel=rel_error_tol)
            for param, errors2 in result.params.items():
                assert errors[param]['lower'] == pytest.approx(
                    errors2['error42']['lower'], rel=rel_error_tol)
                assert errors[param]['upper'] == pytest.approx(
                    errors2['error42']['upper'], rel=rel_error_tol)

            # test custom error
            def custom_error_func(result, params, cl):
                return OrderedDict(
                    (param, {'myval': 42}) for param in params), None

            custom_errors, _ = result.errors(method=custom_error_func,
                                             name='custom_method1')
            for param, errors2 in result.params.items():
                assert custom_errors[param]['myval'] == 42

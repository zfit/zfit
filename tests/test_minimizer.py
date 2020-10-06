#  Copyright (c) 2020 zfit
from collections import OrderedDict

import pytest
import tensorflow as tf
import numpy as np

import zfit.minimizers.baseminimizer as zmin
import zfit.minimizers.optimizers_tf
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


def teardown_function():
    zfit.settings.options['numerical_grad'] = False
    from zfit.core.testing import teardown_function as td_func
    td_func()


from zfit.minimizers.minimizer_tfp import BFGS

true_mu = 4.5
true_sigma = 2
true_lambda = -0.03


def create_loss(obs1):
    mu_param = zfit.Parameter("mu", 4.3, -5., 9.,
                              step_size=0.03)
    sigma_param = zfit.Parameter("sigma", 1.7, 0.01, 10, step_size=0.03)
    lambda_param = zfit.Parameter("lambda", -0.04, -0.5, -0.0003, step_size=0.001)

    gauss1 = zfit.pdf.Gauss(mu=mu_param, sigma=sigma_param, obs=obs1)
    exp1 = zfit.pdf.Exponential(lam=lambda_param, obs=obs1)

    sum_pdf1 = zfit.pdf.SumPDF([gauss1, exp1], 0.8)
    # load params for sampling
    with mu_param.set_value(true_mu):
        with sigma_param.set_value(true_sigma):
            with lambda_param.set_value(true_lambda):
                sampled_data = sum_pdf1.create_sampler(n=30000)
                sampled_data.resample()

                loss = zfit.loss.UnbinnedNLL(model=sum_pdf1, data=sampled_data)
                minimum = loss.value().numpy()

    return loss, minimum, (mu_param, sigma_param, lambda_param)


minimizers = [  # minimizers, minimizer_kwargs, do error estimation
    # (zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)),
    #  False),
    (zfit.minimizers.optimizers_tf.Adam, dict(learning_rate=0.05), False),
    (zfit.minimize.Minuit, {}, True),
    # (BFGS, {}, True),  # TODO: reactivate BFGS!  # check for one not dependent on Minuit
    # (zfit.minimize.Scipy, {}, False),
]

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


def test_dependent_param_extraction():
    obs = zfit.Space("x", limits=(-2, 3))
    mu = zfit.Parameter("mu", 1.2, -4, 6)
    sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)
    sigma1 = zfit.ComposedParameter('sigma1', lambda sigma, mu: sigma + mu, [sigma, mu])
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
    normal_np = np.random.normal(loc=2., scale=3., size=10)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
    minimizer = zfit.minimize.Minuit()
    params_checked = minimizer._check_input_params(nll, params=[mu, sigma1])
    assert {mu, sigma} == set(params_checked)
    sigma.floating = False
    params_checked = minimizer._check_input_params(nll, params=[mu, sigma1])
    assert {mu, } == set(params_checked)


# @pytest.mark.run(order=4)
@pytest.mark.parametrize("chunksize", [3000, 100000])
# skip the numerical gradient due to memory leak bug, TF2.3 fix: https://github.com/tensorflow/tensorflow/issues/35010
@pytest.mark.parametrize("num_grad", [bo for bo in [False, True] if not bo or zfit.run.get_graph_mode()])
@pytest.mark.parametrize("spaces", [obs1, obs1_split])
@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers)
@pytest.mark.flaky(reruns=3)
def test_minimizers(minimizer_class_and_kwargs, num_grad, chunksize, spaces):
    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize
    zfit.settings.options['numerical_grad'] = num_grad

    # minimize_func(minimizer_class_and_kwargs, obs=spaces)
    obs = spaces
    loss, true_minimum, (mu_param, sigma_param, lambda_param) = create_loss(obs1=obs)

    parameter_tolerance = 0.06
    max_distance_to_min = 10.

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)

    # Currently not working, stop the test here. Memory leak?
    if isinstance(minimizer, BFGS) and num_grad and not zfit.run.mode['graph']:
        return

    result = minimizer.minimize(loss=loss)
    cur_val = loss.value().numpy()
    aval, bval, cval = [zfit.run(v) for v in (mu_param, sigma_param, lambda_param)]

    assert true_minimum == pytest.approx(cur_val, abs=max_distance_to_min)
    assert true_mu == pytest.approx(aval, abs=parameter_tolerance)
    assert true_sigma == pytest.approx(bval, abs=parameter_tolerance)
    assert true_lambda == pytest.approx(cval, abs=parameter_tolerance)
    assert result.converged

    # Test Hesse
    if test_error:
        methods = ['hesse_np']
        if isinstance(minimizer, zfit.minimize.Minuit):
            methods.append('minuit_hesse')
        for method in methods:
            sigma_hesse = result.hesse(params=sigma_param, method=method)
            assert tuple(sigma_hesse.keys()) == (sigma_param,)
            errors = result.hesse()
            sigma_hesse = sigma_hesse[sigma_param]
            assert abs(sigma_hesse['error']) == pytest.approx(0.0965, abs=0.15)
            assert abs(errors[sigma_param]['error']) == pytest.approx(0.0965, abs=0.15)
            assert abs(errors[lambda_param]['error']) == pytest.approx(0.01, abs=0.01)

        if isinstance(minimizer, zfit.minimize.Minuit):
            # Test Error
            a_errors, _ = result.errors(params=mu_param)
            assert tuple(a_errors.keys()) == (mu_param,)
            errors, _ = result.errors()
            a_error = a_errors[mu_param]
            assert a_error['lower'] == pytest.approx(-a_error['upper'], abs=0.1)
            assert abs(a_error['lower']) == pytest.approx(0.015, abs=0.015)
            assert abs(errors[sigma_param]['lower']) == pytest.approx(0.010, abs=0.01)
            assert abs(errors[lambda_param]['lower']) == pytest.approx(0.007, abs=0.15)
            assert abs(errors[lambda_param]['upper']) == pytest.approx(0.007, abs=0.15)

            assert errors[mu_param]['lower'] == pytest.approx(a_error['lower'], rel=0.01)
            assert errors[mu_param]['upper'] == pytest.approx(a_error['upper'], rel=0.01)

            # Test Error method name
            a_errors, _ = result.errors(params=mu_param, error_name='error1')
            assert tuple(a_errors.keys()) == (mu_param,)
            errors, _ = result.errors(error_name='error42')
            a_error = a_errors[mu_param]

            assert a_error['lower'] == pytest.approx(result.params[mu_param]['error42']['lower'], rel=0.001)
            assert a_error['lower'] == pytest.approx(result.params[mu_param]['error1']['lower'], rel=0.001)
            for param, errors2 in result.params.items():
                assert errors[param]['lower'] == pytest.approx(errors2['error42']['lower'], rel=0.001)
                assert errors[param]['upper'] == pytest.approx(errors2['error42']['upper'], rel=0.001)

            # test custom error
            def custom_error_func(result, params, sigma):
                return OrderedDict((param, {'myval': 42}) for param in params), None

            custom_errors, _ = result.errors(method=custom_error_func, error_name='custom_method1')
            for param, errors2 in result.params.items():
                assert custom_errors[param]['myval'] == 42

            # Test Hesse

            for method in ['minuit_hesse', 'hesse_np']:
                b_hesses = result.hesse(params=sigma_param, method=method)
                assert tuple(b_hesses.keys()) == (sigma_param,)
                errors = result.hesse()
                b_hesse = b_hesses[sigma_param]
                assert abs(b_hesse['error']) == pytest.approx(0.0965, abs=0.105)
                assert abs(b_hesse['error']) == pytest.approx(abs(errors[sigma_param]['error']), rel=0.1)
                assert abs(errors[sigma_param]['error']) == pytest.approx(0.010, abs=0.015)
                assert abs(errors[lambda_param]['error']) == pytest.approx(0.007, abs=0.015)

    else:
        with pytest.raises(TypeError):
            _ = result.errors(params=mu_param, method="minuit_minos")

#  Copyright (c) 2019 zfit
import pytest
from zfit.core.testing import setup_function, teardown_function, tester

import tensorflow as tf
import zfit
from zfit import ztf
import numpy as np

true_a = 1.
true_b = 4.
true_c = -0.3


def create_loss():
    with tf.variable_scope("func1"):
        a_param = zfit.Parameter("variable_a15151", 1.5, -1., 20.,
                                 step_size=ztf.constant(0.1))
        b_param = zfit.Parameter("variable_b15151", 3.5)
        c_param = zfit.Parameter("variable_c15151", -0.04)
        obs1 = zfit.Space(obs='obs1', limits=(-2.4, 9.1))

        # load params for sampling
        a_param.load(true_a)
        b_param.load(true_b)
        c_param.load(true_c)

    gauss1 = zfit.pdf.Gauss(mu=a_param, sigma=b_param, obs=obs1)
    exp1 = zfit.pdf.Exponential(lambda_=c_param, obs=obs1)

    sum_pdf1 = 0.9 * gauss1 + exp1

    sampled_data = sum_pdf1.create_sampler(n=15000)
    sampled_data.resample()

    loss = zfit.loss.UnbinnedNLL(model=sum_pdf1, data=sampled_data, fit_range=obs1)

    return loss, (a_param, b_param, c_param)


def create_fitresult(minimizer_class_and_kwargs):
    loss, (a_param, b_param, c_param) = create_loss()

    true_minimum = zfit.run(loss.value())

    parameter_tolerance = 0.25  # percent
    max_distance_to_min = 10.

    for param in [a_param, b_param, c_param]:
        zfit.run(param.initializer)  # reset the value

    minimizer_class, minimizer_kwargs, test_error = minimizer_class_and_kwargs
    minimizer = minimizer_class(**minimizer_kwargs)

    result = minimizer.minimize(loss=loss)
    cur_val = zfit.run(loss.value())
    aval, bval, cval = zfit.run([v for v in (a_param, b_param, c_param)])

    ret = {'result': result, 'true_min': true_minimum, 'cur_val': cur_val, 'a': aval, 'b': bval, 'c': cval,
           'a_param': a_param, 'b_param': b_param, 'c_param': c_param}

    return ret


minimizers = [
    # (zfit.minimize.WrapOptimizer, dict(optimizer=tf.train.AdamOptimizer(learning_rate=0.5)), False),
    # (zfit.minimize.Adam, dict(learning_rate=0.5), False),
    (zfit.minimize.Minuit, {}, True),
    # (zfit.minimize.Scipy, {}, False),
]


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers)
def test_fmin(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results['result']
    assert pytest.approx(results['cur_val']) == result.fmin


@pytest.mark.parametrize("minimizer_class_and_kwargs", minimizers)
def test_covariance(minimizer_class_and_kwargs):
    results = create_fitresult(minimizer_class_and_kwargs=minimizer_class_and_kwargs)
    result = results['result']
    hesse = result.hesse()
    a = results['a_param']
    b = results['b_param']
    c = results['c_param']

    cov_mat_3 = result.covariance(params=[a, b, c])
    cov_mat_2 = result.covariance(params=[c, b])
    cov_dict = result.covariance(params=[a, b, c], as_dict=True)

    assert pytest.approx(hesse[a]['error'], rel=0.01) == np.sqrt(cov_dict[(a, a)])
    assert pytest.approx(hesse[a]['error'], rel=0.01) == np.sqrt(cov_mat_3[0, 0])

    assert pytest.approx(hesse[b]['error'], rel=0.01) == np.sqrt(cov_dict[(b, b)])
    assert pytest.approx(hesse[b]['error'], rel=0.01) == np.sqrt(cov_mat_3[1, 1])
    assert pytest.approx(hesse[b]['error'], rel=0.01) == np.sqrt(cov_mat_2[1, 1])

    assert pytest.approx(hesse[c]['error'], rel=0.01) == np.sqrt(cov_dict[(c, c)])
    assert pytest.approx(hesse[c]['error'], rel=0.01) == np.sqrt(cov_mat_3[2, 2])
    assert pytest.approx(hesse[c]['error'], rel=0.01) == np.sqrt(cov_mat_2[0, 0])

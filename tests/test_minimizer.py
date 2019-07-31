#  Copyright (c) 2019 zfit

from collections import OrderedDict
import functools
import time

import pytest
import tensorflow as tf
import numpy as np

from zfit.core.loss import SimpleLoss
import zfit.minimizers.baseminimizer as zmin
from zfit import ztf
import zfit.minimizers.optimizers_tf
from zfit.core.testing import setup_function, teardown_function, tester

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


def minimize_func(minimizer_class_and_kwargs):
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

    assert true_a == pytest.approx(aval, abs=parameter_tolerance)
    assert true_b == pytest.approx(bval, abs=parameter_tolerance)
    assert true_c == pytest.approx(cval, abs=parameter_tolerance)
    assert true_minimum == pytest.approx(cur_val, abs=max_distance_to_min)

    if test_error:
        # Test Error
        a_errors = result.error(params=a_param)
        assert tuple(a_errors.keys()) == (a_param,)
        errors = result.error()
        a_error = a_errors[a_param]
        assert a_error['lower'] == pytest.approx(-a_error['upper'], abs=0.1)
        assert abs(a_error['lower']) == pytest.approx(0.17, abs=0.15)
        assert abs(errors[b_param]['lower']) == pytest.approx(0.17, abs=0.15)
        assert abs(errors[c_param]['lower']) == pytest.approx(0.04, abs=0.15)
        assert abs(errors[c_param]['upper']) == pytest.approx(0.04, abs=0.15)

        assert errors[a_param]['lower'] == pytest.approx(a_error['lower'], rel=0.01)
        assert errors[a_param]['upper'] == pytest.approx(a_error['upper'], rel=0.01)

        # Test Error method name
        a_errors = result.error(params=a_param, error_name='error1')
        assert tuple(a_errors.keys()) == (a_param,)
        errors = result.error(error_name='error42')
        a_error = a_errors[a_param]

        assert a_error['lower'] == pytest.approx(result.params[a_param]['error42']['lower'], rel=0.001)
        assert a_error['lower'] == pytest.approx(result.params[a_param]['error1']['lower'], rel=0.001)
        for param, errors2 in result.params.items():
            assert errors[param]['lower'] == pytest.approx(errors2['error42']['lower'], rel=0.001)
            assert errors[param]['upper'] == pytest.approx(errors2['error42']['upper'], rel=0.001)

        # test custom error
        def custom_error_func(result, params, sigma):
            return OrderedDict((param, {'myval': 42}) for param in params)

        custom_errors = result.error(method=custom_error_func, error_name='custom_method1')
        for param, errors2 in result.params.items():
            assert custom_errors[param]['myval'] == 42

        # Test Hesse
        b_hesses = result.hesse(params=b_param)
        assert tuple(b_hesses.keys()) == (b_param,)
        errors = result.hesse()
        b_hesse = b_hesses[b_param]
        assert abs(b_hesse['error']) == pytest.approx(0.0965, abs=0.15)
        assert abs(errors[b_param]['error']) == pytest.approx(0.0965, abs=0.15)
        assert abs(errors[c_param]['error']) == pytest.approx(0.1, abs=0.15)

    else:
        with pytest.raises(TypeError):
            _ = result.error(params=a_param)


minimizers = [  # minimizers, minimizer_kwargs, do error estimation
    (zfit.minimizers.optimizers_tf.WrapOptimizer, dict(optimizer=tf.train.AdamOptimizer(learning_rate=0.5)), False),
    (zfit.minimizers.optimizers_tf.Adam, dict(learning_rate=0.5), False),
    (zfit.minimize.Minuit, {}, True),
    (zfit.minimize.Scipy, {}, False),
]


@pytest.mark.parametrize("chunksize", [10000000, 3000])
@pytest.mark.parametrize("minimizer_class", minimizers)
@pytest.mark.flaky(reruns=3)
def test_minimizers(minimizer_class, chunksize):
    zfit.run.chunking.active = True
    zfit.run.chunking.max_n_points = chunksize
    minimize_func(minimizer_class)

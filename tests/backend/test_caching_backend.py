#  Copyright (c) 2024 zfit
import pytest
import tensorflow as tf

import zfit
from zfit import z


# this works now also for tensorflow, as all variables given as args, effectively Params, have a
# custom tf retracing behavior on their identity
# Therefore, the legacy fix in z.function works (still, but z.function is not needed anymore for this)
# and with tf.function
@pytest.mark.parametrize("function", [z.function, tf.function])
def test_parameter_caching(function):
    # this is to ensure that we fixed the bug (https://github.com/tensorflow/tensorflow/issues/57365) internally

    x1 = zfit.Parameter("x1", 2.0)
    x2 = zfit.Parameter("x2", 4.0)


    def one_plus(x):
        return x + 1

    cx1 = zfit.ComposedParameter("cx1", one_plus, params=[x1])
    cx2 = zfit.ComposedParameter("cx2", one_plus, params=[x2])

    ncompile1 = 0
    ncompile2 = 0


    @tf.function(autograph=False)
    def f():
        nonlocal ncompile1
        ncompile1 += 1
        res = x1 + x2**2 / 2
        return res

    def f2():
        nonlocal ncompile2
        ncompile2 += 1
        res = cx1 + cx2**2 / 2
        return res

    ncompgrad1 = 0
    ncompgrad2 = 0
    def grad(param):
        nonlocal ncompgrad1
        ncompgrad1 += 1

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(param)
            value = f()
        return tf.stack(tape.gradient(value, param))

    def grad2(param):
        nonlocal ncompgrad2
        ncompgrad2 += 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(param)
            value = f2()
        return tf.stack(tape.gradient(value, param))

    jitted_grad = function(grad)
    jitted_grad2 = function(grad2)

    y1 = grad(x1)
    ncomp1_after1 = ncompile1
    y1_jit = jitted_grad(x1)
    ncompgrad1_after1 = ncompgrad1
    assert ncomp1_after1 == ncompile1
    y1_jit = jitted_grad(x1)
    assert ncompgrad1_after1 == ncompgrad1
    with x1.set_value(142.):
        y1_jit = jitted_grad(x1)
        assert ncompgrad1_after1 == ncompgrad1

    y1_jit = jitted_grad(x1)
    assert ncompgrad1_after1 == ncompgrad1

    assert abs(y1 - 1.0) < 1e-5  # because d x1 / dx1 = 1
    assert abs(y1_jit - 1.0) < 1e-5
    y2 = grad(x2)
    y2_jit = jitted_grad(x2)

    assert abs(y2 - 4.0) < 1e-5  # because d / dx x**2/2 = x -> 4
    assert abs(y2_jit - y2) < 1e-5
    assert abs(y2_jit - 4.0) < 1e-5

    # use both parameters
    y = grad([x1, x2])
    ncompgrad1_after2 = ncompgrad1
    assert ncompgrad1_after2 > ncompgrad1_after1
    y_jit = jitted_grad([x1, x2])

    assert abs(y[0] - 1.0) < 1e-5
    assert abs(y[1] - 4.0) < 1e-5
    assert abs(y_jit[0] - 1.0) < 1e-5
    assert abs(y_jit[1] - 4.0) < 1e-5

    # use both parameters, swap order
    y = grad([x2, x1])

    y_jit = jitted_grad([x2, x1])

    assert pytest.approx(y[0], 1e-5) == 4.0
    assert pytest.approx(y[1], 1e-5) == 1.0
    assert pytest.approx(y_jit[0], 1e-5) == 4.0
    assert pytest.approx(y_jit[1], 1e-5) == 1.0

    # test with composed parameters
    y = grad2([x1, x2])
    y_jit = jitted_grad2([x1, x2])

    assert pytest.approx(y[0], 1e-5) == y_jit[0]
    assert pytest.approx(y[1], 1e-5) == y_jit[1]

    # test with composed parameters, swap order
    y = grad2([x2, x1])
    y_jit = jitted_grad2([x2, x1])

    assert pytest.approx(y[0], 1e-5) == y_jit[0]
    assert pytest.approx(y[1], 1e-5) == y_jit[1]

def test_same_params_dont_fail():

    low, high = 0, 1
    obs = zfit.Space('obs', low, high)

    sigma = zfit.Parameter('sigma', 2.)
    alpha = zfit.Parameter('alpha', 5.)

    pdf = zfit.pdf.GeneralizedGaussExpTail(
        obs=obs,
        mu=0,
        sigmar=sigma,
        sigmal=sigma,
        alphar=alpha,
        alphal=alpha
    )
    value = pdf.integrate([low, high])  # should not raise an error
    assert pytest.approx(value.numpy(), rel=1e-5) == 1.0

    @tf.function(autograph=False)
    def test(x, y):
        return x + y

    var1 =  tf.Variable(1.0, name='var1')
    var2 =  tf.Variable(3.0, name='var2')

    value = test(var1, var2)
    assert value == 4.0
    value = test(var2, var2)
    assert value == 6.0
    value = test(sigma, alpha)
    assert value == 7.0
    value = test(sigma, sigma)  # this should not raise an error
    assert value == 4.0


    @z.function(autograph=False)
    def testz(x, y):
        return x + y

    value = testz(var1, var2)
    assert value == 4.0
    value = testz(var2, var2)
    assert value == 6.0
    value = testz(sigma, alpha)
    assert value == 7.0
    value = testz(sigma, sigma)  # this should not raise an error

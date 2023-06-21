#  Copyright (c) 2023 zfit
import pytest

import zfit


def test_parameter_caching():
    zfit.run.set_graph_mode(False)
    # this is to ensure that we fixed the bug (https://github.com/tensorflow/tensorflow/issues/57365) internally
    import tensorflow as tf
    from zfit import z

    x1 = zfit.Parameter("x1", 2.0)
    x2 = zfit.Parameter("x2", 4.0)

    def one_plus(x):
        return x + 1

    cx1 = zfit.ComposedParameter("cx1", one_plus, dependents=[x1])
    cx2 = zfit.ComposedParameter("cx2", one_plus, dependents=[x2])

    @tf.function(autograph=False)
    def f():
        res = x1 + x2**2 / 2
        return res

    def f2():
        res = cx1 + cx2**2 / 2
        return res

    def grad(param):
        param = zfit.z.math._extract_tfparams(param)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(param)
            value = f()
        return tf.stack(tape.gradient(value, param))

    def grad2(param):
        param = zfit.z.math._extract_tfparams(param)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(param)
            value = f2()
        return tf.stack(tape.gradient(value, param))

    jitted_grad = z.function(grad)
    jitted_grad2 = z.function(grad2)
    # jitted_grad = tf.function(grad)

    y1 = grad(x1)
    y1_jit = jitted_grad(x1)
    assert abs(y1 - 1.0) < 1e-5  # because d x1 / dx1 = 1
    assert abs(y1_jit - 1.0) < 1e-5
    y2 = grad(x2)
    y2_jit = jitted_grad(x2)

    assert abs(y2 - 4.0) < 1e-5  # because d / dx x**2/2 = x -> 4
    assert abs(y2_jit - y2) < 1e-5
    assert abs(y2_jit - 4.0) < 1e-5

    # use both parameters
    y = grad([x1, x2])
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

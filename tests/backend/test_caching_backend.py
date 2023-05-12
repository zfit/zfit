#  Copyright (c) 2023 zfit
import zfit


def test_parameter_caching():
    import tensorflow as tf

    x1 = zfit.Parameter("x1", 2.0)
    x2 = zfit.Parameter("x2", 4.0)

    def f():
        res = x1 + x2**2 / 2
        return res

    def grad(param):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(param)
            value = f()
        return tape.gradient(value, param)

    jitted_grad = tf.function(grad)

    y1 = grad(x1)
    y1_jit = jitted_grad(x1)
    assert abs(y1 - 1.0) < 1e-5  # because d x1 / dx1 = 1
    assert abs(y1_jit - 1.0) < 1e-5
    y2 = grad(x2)
    y2_jit = jitted_grad(x2)
    print(f"y2: {y2}, should be 4")  # but is 1 because it uses the derivative of x1
    print(
        f"y2_jit: {y2_jit}, should also be 4"
    )  # but is 1 because it uses the derivative of x1
    assert abs(y2 - 4.0) < 1e-5  # because d / dx x**2/2 = x -> 4
    assert abs(y2_jit - 4.0) < 1e-5  # fails!

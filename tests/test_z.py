#  Copyright (c) 2022 zfit

import numpy as np
import tensorflow as tf

from zfit import z


def numpy_func(x, a):
    return np.square(x) * a


@z.function
def wrapped_numpy_func(x_tensor, a_tensor):
    result = tf.numpy_function(
        func=numpy_func, inp=[x_tensor, a_tensor], Tout=tf.float64
    )
    result = tf.sqrt(result)
    return result


def test_wrapped_func():
    rnd = z.random.uniform(shape=(10,))
    result = wrapped_numpy_func(rnd, z.constant(3.0))
    np.testing.assert_allclose(rnd * np.sqrt(3), result)


def test_multinomial():
    import zfit.z.numpy as znp

    probs = np.array([0.5, 0.5])
    x = z.random.counts_multinomial(10000, probs=probs)
    assert x.shape == (2,)
    probs = np.array([0.1, 0.2, 0.7])
    x = z.random.counts_multinomial(10000, probs=probs)
    assert x.shape == (3,)
    probs = np.array([0.5, 0.5])
    x = z.random.counts_multinomial(10000, probs=probs, dtype=znp.float64)
    assert x.shape == (2,)
    probs = np.array([0.1, 0.7, 0.2])
    x = z.random.counts_multinomial(10000, probs=probs, dtype=znp.float64)
    assert x.shape == (3,)

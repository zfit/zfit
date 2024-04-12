#  Copyright (c) 2022 zfit
import numpy as np
import tensorflow as tf
from scipy.special import wofz

from zfit import z
import zfit.z.numpy as znp


def test_z_numpy_ndarray_is_tensorflow_tensor():
    """In tensorflow 2.4.1 tf.experimental.numpy.ndarray was a wrapper around tf.Tensor.

    Now this concept seems to have been scratched and tf.experimental.numpy.ndarray is just an alias for tf.Tensor. See
    the commit history of
    https://github.com/tensorflow/tensorflow/commits/master/tensorflow/python/ops/numpy_ops/np_arrays.py
    """
    assert znp.ndarray is tf.Tensor
    assert isinstance(znp.array(1), tf.Tensor)
    assert isinstance(znp.sum(znp.array(0)), tf.Tensor)


def test_faddeeva():
    """We only test with positive imaginary part as the Humlicek approximation of the Faddeeva function is becoming
    unstable in the negative imaginary part region.

    That is also due to the nature of the Faddeeva function in that region.
    """
    size = int(1e4)
    test_values1 = np.random.uniform(0, 0.1, size) + 1.0j * np.random.uniform(
        0, 0.1, size
    )
    test_values2 = np.random.uniform(0, 1, size) + 1.0j * np.random.uniform(0, 1, size)
    test_values3 = np.random.uniform(0, 10, size) + 1.0j * np.random.uniform(
        0, 10, size
    )
    test_values4 = np.random.uniform(0, 100, size) + 1.0j * np.random.uniform(
        0, 100, size
    )
    test_values5 = np.random.uniform(-0.1, 0, size) + 1.0j * np.random.uniform(
        0, 1, size
    )
    test_values6 = np.random.uniform(-1, 0, size) + 1.0j * np.random.uniform(0, 1, size)
    test_values7 = np.random.uniform(-10, 0, size) + 1.0j * np.random.uniform(
        0, 10, size
    )
    test_values8 = np.random.uniform(-100, 0, size) + 1.0j * np.random.uniform(
        0, 100, size
    )
    test_values = np.concatenate(
        [
            test_values1,
            test_values2,
            test_values3,
            test_values4,
            test_values5,
            test_values6,
            test_values7,
            test_values8,
        ]
    )
    np.testing.assert_allclose(
        znp.faddeeva_humlicek(test_values), wofz(test_values), rtol=1e-06
    )

    test_values_tensor = znp.asarray(test_values)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(test_values_tensor)
        result = znp.faddeeva_humlicek(test_values_tensor)
    gradientstf = tape.gradient(result, test_values_tensor)
    assert np.all(np.isfinite(gradientstf))

    # we test on a small sample of size 10 because the numerical gradient is slow to compute
    # we restrict our selves to real x and not go far away from 0 (-1, 1) because the Faddeeva function flattens out and gradients become zero.
    # we test the real and imaginary part separately because numdifftools can't differentiate with respect to complex numbers.
    x = np.random.uniform(-1, 1, size=10)
    f_real = lambda x: znp.real(znp.faddeeva_humlicek(x))
    f_imag = lambda x: znp.imag(znp.faddeeva_humlicek(x))

    num_gradients_real = z.math.numdifftools.Gradient(f_real)(x)
    num_gradients_imag = z.math.numdifftools.Gradient(f_imag)(x)

    x = znp.asarray(x)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        result_real = f_real(x)
        result_imag = f_imag(x)
    tf_gradients_real = tape.gradient(result_real, x)
    tf_gradients_imag = tape.gradient(result_imag, x)

    np.testing.assert_allclose(np.diag(num_gradients_real), tf_gradients_real)
    np.testing.assert_allclose(np.diag(num_gradients_imag), tf_gradients_imag)

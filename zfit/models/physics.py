import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import numpy as np

import zfit
from zfit import ztf


def powerlaw(x, a, k):
    return a * x ** (k)


def gauss_func(x, mu, sigma):
    exp_term = tf.exp((x - mu / sigma) ** 2 / 2)
    return exp_term


def crystalball_func(x, mean, sigma, alpha, n):
    t = (x - mean) / sigma
    # t = tf.where(tf.greater_equal(alpha, 0.), t, -t)
    t *= tf.sign(alpha)
    abs_alpha = tf.abs(alpha)
    A = powerlaw(n / abs_alpha, 1., n) * tf.exp(- abs_alpha ** 2 / 2)
    B = powerlaw(n / abs_alpha, 1., 1) - abs_alpha
    cond = tf.greater_equal(t, -abs_alpha)
    func = tf.where(cond, gauss_func(x=x, mu=mean, sigma=sigma), powerlaw(B - t, A, -n))

    if False:
        data_norm = tf.random_uniform(shape=(normsize,), minval=xmin, maxval=xmax, dtype=tf.float64)
        func_intg = numeric_integral(func, xmin, xmax)
        return fun
        c / func_intg
    else:
        return func


if __name__ == '__main__':
    mu, sigma, alpha, n = [ztf.constant(1.) for _ in range(4)]
    res = crystalball_func(np.random.random(size=100), mu, sigma, alpha, n)

    print(zfit.run(res))

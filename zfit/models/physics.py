import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import numpy as np

import zfit
from zfit import ztf


def powerlaw(x, a, k):
    return a * x ** k


def crystalball_func(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * tf.sign(alpha)
    # t = tf.where(tf.greater_equal(alpha, 0.), t, -t)
    # t *= tf.sign(alpha)
    abs_alpha = tf.abs(alpha)
    A = (n / abs_alpha) ** n * tf.exp(- 0.5 * abs_alpha ** 2)
    B = (n / abs_alpha) - abs_alpha
    cond = tf.greater_equal(t, -abs_alpha)
    func = tf.where(cond, tf.exp(0.5 * t ** 2), powerlaw(B - t, A, -n))

    return func

def crystalball_integral(limits, params):
    pass


if __name__ == '__main__':
    mu, sigma, alpha, n = [ztf.constant(1.) for _ in range(4)]
    res = crystalball_func(np.random.random(size=100), mu, sigma, alpha, n)

    print(zfit.run(res))

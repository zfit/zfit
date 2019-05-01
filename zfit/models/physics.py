#  Copyright (c) 2019 zfit

from typing import Type, Any

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import numpy as np

import zfit
from zfit import ztf
from ..core.basepdf import BasePDF
from ..core.limits import ANY_UPPER, ANY_LOWER, Space
from ..settings import ztypes
from ..util import ztyping


def _powerlaw(x, a, k):
    return a * tf.pow(x, k)


def crystalball_func(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = tf.abs(alpha)
    a = tf.pow((n / abs_alpha), n) * tf.exp(-0.5 * tf.square(alpha))
    b = (n / abs_alpha) - abs_alpha
    cond = tf.less(t, -abs_alpha)
    # func = tf.where(cond, tf.exp(-0.5 * tf.square(t)), _powerlaw(b - t, a, -n))
    func = ztf.safe_where(cond,
                          lambda t: _powerlaw(b - t, a, -n),
                          lambda t: tf.exp(-0.5 * tf.square(t)),
                          values=t, value_safer=lambda t: tf.ones_like(t) * (b - 2))

    return func


def double_crystalball_func(x, mu, sigma, alphal, nl, alphar, nr):
    cond = tf.less(x, mu)
    func = tf.where(cond,
                    crystalball_func(x, mu, sigma, alphal, nl),
                    crystalball_func(x, mu, sigma, -alphar, nr))

    return func


# def _python_crystalball_integral(limits, params):  # not working with tf, used for autoconvert
#     mu = params['mu']
#     sigma = params['sigma']
#     alpha = params['alpha']
#     n = params['n']
#
#     (lower,), (upper,) = limits.limits
#
#     sqrt_pi_over_two = np.sqrt(np.pi / 2)
#     sqrt2 = np.sqrt(2)
#
#     result = 0.0
#     use_log = tf.abs(n - 1.0) < 1.0e-05
#
#     abs_sigma = tf.abs(sigma)
#     abs_alpha = tf.abs(alpha)
#
#     tmin = (lower - mu) / abs_sigma
#     tmax = (upper - mu) / abs_sigma
#
#     if alpha < 0:
#         tmin, tmax = -tmax, -tmin
#
#     if tmin >= -abs_alpha:
#         result += abs_sigma * sqrt_pi_over_two * (tf.erf(tmax / sqrt2)
#                                                   - tf.erf(tmin / sqrt2))
#
#     elif tmax <= -abs_alpha:
#         a = tf.pow(n / abs_alpha, n) * tf.exp(-0.5 * tf.square(abs_alpha))
#
#         b = n / abs_alpha - abs_alpha
#
#         if use_log:
#             result += a * abs_sigma * (tf.log(b - tmin) - tf.log(b - tmax))
#         else:
#             result += a * abs_sigma / (1.0 - n) * (1.0 / (tf.pow(b - tmin, n - 1.0))
#                                                    - 1.0 / (tf.pow(b - tmax, n - 1.0)))
#     else:
#         a = tf.pow(n / abs_alpha, n) * tf.exp(-0.5 * tf.square(abs_alpha))
#         b = n / abs_alpha - abs_alpha
#
#         if use_log:
#             term1 = a * abs_sigma * (tf.log(b - tmin) - tf.log(n / abs_alpha))
#
#         else:
#             term1 = a * abs_sigma / (1.0 - n) * (1.0 / (tf.pow(b - tmin, n - 1.0))
#                                                  - 1.0 / (tf.pow(n / abs_alpha, n - 1.0)))
#
#         term2 = abs_sigma * sqrt_pi_over_two * (tf.erf(tmax / sqrt2)
#                                                 - tf.erf(-abs_alpha / sqrt2))
#
#         result += term1 + term2
#
#     return result

# created with the help of TensorFlow autograph used on python code converted from ShapeCB of RooFit

def crystalball_integral(limits, params, model):
    mu = params['mu']
    sigma = params['sigma']
    alpha = params['alpha']
    n = params['n']

    (lower,), (upper,) = limits.limits
    lower = lower[0]  # obs number 0
    upper = upper[0]

    sqrt_pi_over_two = np.sqrt(np.pi / 2)
    sqrt2 = np.sqrt(2)
    result = 0.0

    use_log = tf.less(tf.abs(n - 1.0), 1e-05)
    abs_sigma = tf.abs(sigma)
    abs_alpha = tf.abs(alpha)

    tmin = (lower - mu) / abs_sigma
    tmax = (upper - mu) / abs_sigma

    def if_true():
        return tf.negative(tmin), tf.negative(tmax)

    def if_false():
        return tmax, tmin

    tmax, tmin = tf.cond(tf.less(alpha, 0), if_true, if_false)

    def if_true_4():
        result_5, = result,
        result_5 += abs_sigma * sqrt_pi_over_two * (tf.erf(tmax / sqrt2) - tf.erf(tmin / sqrt2))
        return result_5

    def if_false_4():
        result_6 = result

        def if_true_3():
            result_3 = result_6
            a = tf.pow(n / abs_alpha, n) * tf.exp(-0.5 * tf.square(abs_alpha))
            b = n / abs_alpha - abs_alpha

            def if_true_1():
                result_1, = result_3,
                result_1 += a * abs_sigma * (tf.log(b - tmin) - tf.log(b - tmax))
                return result_1

            def if_false_1():
                result_2, = result_3,
                result_2 += a * abs_sigma / (1.0 - n) * (
                    1.0 / tf.pow(b - tmin, n - 1.0) - 1.0 / tf.pow(b - tmax, n - 1.0))
                return result_2

            result_3 = tf.cond(use_log, if_true_1, if_false_1)
            return result_3

        def if_false_3():
            result_4, = result_6,
            a = tf.pow(n / abs_alpha, n) * tf.exp(-0.5 * tf.square(abs_alpha))
            b = n / abs_alpha - abs_alpha

            def if_true_2():
                term1 = a * abs_sigma * (tf.log(b - tmin) - tf.log(n / abs_alpha))
                return term1

            def if_false_2():
                term1 = a * abs_sigma / (1.0 - n) * (
                    1.0 / tf.pow(b - tmin, n - 1.0) - 1.0 / tf.pow(n / abs_alpha, n - 1.0))
                return term1

            term1 = tf.cond(use_log, if_true_2, if_false_2)
            term2 = abs_sigma * sqrt_pi_over_two * (
                tf.erf(tmax / sqrt2) - tf.erf(-abs_alpha / sqrt2))
            result_4 += term1 + term2
            return result_4

        result_6 = tf.cond(tf.less_equal(tmax, -abs_alpha), if_true_3, if_false_3)
        return result_6

    # if_false_4()
    result = tf.cond(tf.greater_equal(tmin, -abs_alpha), if_true_4, if_false_4)
    return result


def double_crystalball_integral(limits, params, model):
    mu = params['mu']
    sigma = params['sigma']

    (lower,), (upper,) = limits.limits
    lower = lower[0]  # obs number 0
    upper = upper[0]

    limits_left = Space(limits.obs, (lower, mu))
    limits_right = Space(limits.obs, (mu, upper))
    params_left = dict(mu=mu, sigma=sigma, alpha=params["alphal"],
                       n=params["nl"])
    params_right = dict(mu=mu, sigma=sigma, alpha=-params["alphar"],
                        n=params["nr"])

    left = tf.cond(tf.less(mu, lower), 0., crystalball_integral(limits_left, params_left))
    right = tf.cond(tf.greater(mu, upper), 0., crystalball_integral(limits_right, params_right))

    return left + right


class CrystalBall(BasePDF):
    _N_OBS = 1
    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput,
                 alpha: ztyping.ParamTypeInput, n: ztyping.ParamTypeInput,
                 obs: ztyping.ObsTypeInput, name: str = "CrystalBall", dtype: Type = ztypes.float):
        """`Crystal Ball shaped PDF`__. A combination of a Gaussian with an powerlaw tail.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha, n) =  \\begin{cases} \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & \\mbox{for}\\frac{x - \\mu}{\\sigma} \\geqslant -\\alpha \\newline
            A \\cdot (B - \\frac{x - \\mu}{\\sigma})^{-n}, & \\mbox{for }\\frac{x - \\mu}{\\sigma}
             < -\\alpha \\end{cases}

        with

        .. math::
            A = \\left(\\frac{n}{\\left| \\alpha \\right|}\\right)^n \\cdot
            \\exp\\left(- \\frac {\\left|\\alpha \\right|^2}{2}\\right)

            B = \\frac{n}{\\left| \\alpha \\right|}  - \\left| \\alpha \\right|

        Args:
            mu (`zfit.Parameter`): The mean of the gaussian
            sigma (`zfit.Parameter`): Standard deviation of the gaussian
            alpha (`zfit.Parameter`): parameter where to switch from a gaussian to the powertail
            n (`zfit.Parameter`): Exponent of the powertail
            obs (:py:class:`~zfit.Space`):
            name (str):
            dtype (tf.DType):

        .. _CBShape: https://en.wikipedia.org/wiki/Crystal_Ball_function

        __CBShape_
        """
        params = {'mu': mu,
                  'sigma': sigma,
                  'alpha': alpha,
                  'n': n}
        super().__init__(obs=obs, dtype=dtype, name=name, params=params)

    def _unnormalized_pdf(self, x):
        mu = self.params['mu']
        sigma = self.params['sigma']
        alpha = self.params['alpha']
        n = self.params['n']
        x = x.unstack_x()
        return crystalball_func(x=x, mu=mu, sigma=sigma, alpha=alpha, n=n)


crystalball_integral_limits = Space.from_axes(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
# TODO uncomment, dependency: bug in TF (31.1.19) # 25339 that breaks gradient of resource var in cond
# CrystalBall.register_analytic_integral(func=crystalball_integral, limits=crystalball_integral_limits)


class DoubleCB(BasePDF):
    _N_OBS = 1

    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput,
                 alphal: ztyping.ParamTypeInput, nl: ztyping.ParamTypeInput,
                 alphar: ztyping.ParamTypeInput, nr: ztyping.ParamTypeInput,
                 obs: ztyping.ObsTypeInput, name: str = "DoubleCB", dtype: Type = ztypes.float):
        """`Double sided Crystal Ball shaped PDF`__. A combination of two CB using the **mu** (not a frac).
        on each side.

        The function is defined as follows:

        .. math::
            f(x;\\mu, \\sigma, \\alpha_{L}, n_{L}, \\alpha_{R}, n_{R}) =  \\begin{cases}
            A_{L} \\cdot (B_{L} - \\frac{x - \\mu}{\\sigma})^{-n},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma} < -\\alpha_{L} \\newline
            \\exp(- \\frac{(x - \\mu)^2}{2 \\sigma^2}),
            & -\\alpha_{L} \\leqslant \\mbox{for}\\frac{x - \\mu}{\\sigma} \\leqslant \\alpha_{R} \\newline
            A_{R} \\cdot (B_{R} - \\frac{x - \\mu}{\\sigma})^{-n},
             & \\mbox{for }\\frac{x - \\mu}{\\sigma} > \\alpha_{R}
            \\end{cases}

        with

        .. math::
            A_{L/R} = \\left(\\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}\\right)^n_{L/R} \\cdot
            \\exp\\left(- \\frac {\\left|\\alpha_{L/R} \\right|^2}{2}\\right)

            B_{L/R} = \\frac{n_{L/R}}{\\left| \\alpha_{L/R} \\right|}  - \\left| \\alpha_{L/R} \\right|

        Args:
            mu (`zfit.Parameter`): The mean of the gaussian
            sigma (`zfit.Parameter`): Standard deviation of the gaussian
            alphal (`zfit.Parameter`): parameter where to switch from a gaussian to the powertail on the left
            side
            nl (`zfit.Parameter`): Exponent of the powertail on the left side
            alphar (`zfit.Parameter`): parameter where to switch from a gaussian to the powertail on the right
            side
            nr (`zfit.Parameter`): Exponent of the powertail on the right side
            obs (:py:class:`~zfit.Space`):
            name (str):
            dtype (tf.DType):

        """
        params = {'mu': mu,
                  'sigma': sigma,
                  'alphal': alphal,
                  'nl': nl,
                  'alphar': alphar,
                  'nr': nr}
        super().__init__(obs=obs, dtype=dtype, name=name, params=params)

    def _unnormalized_pdf(self, x):
        mu = self.params['mu']
        sigma = self.params['sigma']
        alphal = self.params['alphal']
        nl = self.params['nl']
        alphar = self.params['alphar']
        nr = self.params['nr']
        x = x.unstack_x()
        return double_crystalball_func(x=x, mu=mu, sigma=sigma, alphal=alphal, nl=nl,
                                       alphar=alphar, nr=nr)

# DoubleCB.register_analytic_integral(func=double_crystalball_integral, limits=crystalball_integral_limits)


if __name__ == '__main__':
    mu = ztf.constant(0)
    sigma = ztf.constant(0.5)
    alpha = ztf.constant(3)
    n = ztf.constant(1)
    # res = crystalball_func(np.random.random(size=100), mu, sigma, alpha, n)
    # int1 = crystalball_integral(limits=zfit.Space(obs='obs1', limits=(-3, 5)),
    #                             params={'mu': mu, "sigma": sigma, "alpha": alpha, "n": n})
    from tensorflow.contrib import autograph
    import matplotlib.pyplot as plt

    new_code = autograph.to_code(crystalball_integral)
    obs = zfit.Space(obs='obs1', limits=(-3, 1))
    cb1 = CrystalBall(mu, sigma, alpha, n, obs=obs)
    res = cb1.pdf(np.random.random(size=100))
    int1 = cb1.integrate(limits=(-0.01, 2), norm_range=obs)
    # tf.add_check_numerics_ops()

    x = np.linspace(-5, 1, num=1000)
    vals = cb1.pdf(x=x)
    y = zfit.run(vals)[0]
    plt.plot(x, y)
    plt.show()

    # print(new_code)
    print(zfit.run(res))
    print(zfit.run(int1))

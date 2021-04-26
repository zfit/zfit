#  Copyright (c) 2021 zfit

from typing import Type

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.space import ANY_LOWER, ANY_UPPER, Space
from ..settings import ztypes
from ..util import ztyping


def _powerlaw(x, a, k):
    return a * znp.power(x, k)


@z.function(wraps='zfit_tensor')
def crystalball_func(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = znp.abs(alpha)
    a = znp.power((n / abs_alpha), n) * znp.exp(-0.5 * tf.square(alpha))
    b = (n / abs_alpha) - abs_alpha
    cond = tf.less(t, -abs_alpha)
    func = z.safe_where(cond,
                        lambda t: _powerlaw(b - t, a, -n),
                        lambda t: znp.exp(-0.5 * tf.square(t)),
                        values=t, value_safer=lambda t: tf.ones_like(t) * (b - 2))
    func = znp.maximum(func, tf.zeros_like(func))
    return func


@z.function(wraps='zfit_tensor')
def double_crystalball_func(x, mu, sigma, alphal, nl, alphar, nr):
    cond = tf.less(x, mu)
    func = tf.where(cond,
                    crystalball_func(x, mu, sigma, alphal, nl),
                    crystalball_func(x, mu, sigma, -alphar, nr))

    return func


# created with the help of TensorFlow autograph used on python code converted from ShapeCB of RooFit
def crystalball_integral(limits, params, model):
    mu = params['mu']
    sigma = params['sigma']
    alpha = params['alpha']
    n = params['n']

    lower, upper = limits._rect_limits_tf

    integral = crystalball_integral_func(mu, sigma, alpha, n, lower, upper)
    return integral


# @z.function(wraps='zfit_tensor')
# @tf.function  # BUG? TODO: problem with tf.function and input signature
def crystalball_integral_func(mu, sigma, alpha, n, lower, upper):
    sqrt_pi_over_two = np.sqrt(np.pi / 2)
    sqrt2 = np.sqrt(2)

    use_log = tf.less(znp.abs(n - 1.0), 1e-05)
    abs_sigma = znp.abs(sigma)
    abs_alpha = znp.abs(alpha)
    tmin = (lower - mu) / abs_sigma
    tmax = (upper - mu) / abs_sigma

    def if_true():
        return tf.negative(tmin), tf.negative(tmax)

    def if_false():
        return tmax, tmin

    tmax, tmin = tf.cond(pred=tf.less(alpha, 0), true_fn=if_true, false_fn=if_false)

    def if_true_4():
        return abs_sigma * sqrt_pi_over_two * (tf.math.erf(tmax / sqrt2) - tf.math.erf(tmin / sqrt2))

    def if_false_4():
        result_6 = 0.0

        def if_true_3():
            result_3 = result_6
            a = znp.power(n / abs_alpha, n) * znp.exp(-0.5 * tf.square(abs_alpha))
            b = n / abs_alpha - abs_alpha

            def if_true_1():
                result_1, = result_3,
                result_1 += a * abs_sigma * (znp.log(b - tmin) - znp.log(b - tmax))
                return result_1

            def if_false_1():
                result_2, = result_3,
                result_2 += a * abs_sigma / (1.0 - n) * (
                    1.0 / znp.power(b - tmin, n - 1.0) - 1.0 / znp.power(b - tmax, n - 1.0))
                return result_2

            result_3 = tf.cond(pred=use_log, true_fn=if_true_1, false_fn=if_false_1)
            return result_3

        def if_false_3():
            result_4, = result_6,
            a = znp.power(n / abs_alpha, n) * znp.exp(-0.5 * tf.square(abs_alpha))
            b = n / abs_alpha - abs_alpha

            def if_true_2():
                term1 = a * abs_sigma * (znp.log(b - tmin) - znp.log(n / abs_alpha))
                return term1

            def if_false_2():
                term1 = a * abs_sigma / (1.0 - n) * (
                    1.0 / znp.power(b - tmin, n - 1.0) - 1.0 / znp.power(n / abs_alpha, n - 1.0))
                return term1

            term1 = tf.cond(pred=use_log, true_fn=if_true_2, false_fn=if_false_2)
            term2 = abs_sigma * sqrt_pi_over_two * (
                tf.math.erf(tmax / sqrt2) - tf.math.erf(-abs_alpha / sqrt2))
            result_4 += term1 + term2
            return result_4

        result_6 = tf.cond(pred=tf.less_equal(tmax, -abs_alpha), true_fn=if_true_3, false_fn=if_false_3)
        return result_6

    # if_false_4()
    result = tf.cond(pred=tf.greater_equal(tmin, -abs_alpha), true_fn=if_true_4, false_fn=if_false_4)
    if not result.shape.rank == 0:
        result = tf.gather(result, 0, axis=-1)  # remove last dim, should vanish
    return result


def double_crystalball_mu_integral(limits, params, model):
    mu = params['mu']
    sigma = params['sigma']
    alphal = params["alphal"]
    nl = params["nl"]
    alphar = -params["alphar"]
    nr = params["nr"]

    lower, upper = limits._rect_limits_tf

    return double_crystalball_mu_integral_func(mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr,
                                               lower=lower, upper=upper)


@z.function(wraps='zfit_tensor')
def double_crystalball_mu_integral_func(mu, sigma, alphal, nl, alphar, nr, lower, upper):
    left = tf.cond(pred=tf.less(mu, lower), true_fn=lambda: z.constant(0.),
                   false_fn=lambda: crystalball_integral_func(mu=mu, sigma=sigma, alpha=alphal, n=nl,
                                                              lower=lower, upper=mu))
    right = tf.cond(pred=tf.greater(mu, upper), true_fn=lambda: z.constant(0.),
                    false_fn=lambda: crystalball_integral_func(mu=mu, sigma=sigma, alpha=alphar, n=nr,
                                                               lower=mu, upper=upper))
    integral = left + right
    return integral


class CrystalBall(BasePDF):
    _N_OBS = 1

    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput,
                 alpha: ztyping.ParamTypeInput, n: ztyping.ParamTypeInput,
                 obs: ztyping.ObsTypeInput, name: str = "CrystalBall", dtype: Type = ztypes.float):
        """Crystal Ball shaped PDF. A combination of a Gaussian with a powerlaw tail.

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
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alpha: parameter where to switch from a gaussian to the powertail
            n: Exponent of the powertail
            obs:
            name:
            dtype:

        .. _CBShape: https://en.wikipedia.org/wiki/Crystal_Ball_function
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


crystalball_integral_limits = Space(axes=(0,), limits=(((ANY_LOWER,),), ((ANY_UPPER,),)))
# TODO uncomment, dependency: bug in TF (31.1.19) # 25339 that breaks gradient of resource var in cond
CrystalBall.register_analytic_integral(func=crystalball_integral, limits=crystalball_integral_limits)


class DoubleCB(BasePDF):
    _N_OBS = 1

    def __init__(self, mu: ztyping.ParamTypeInput, sigma: ztyping.ParamTypeInput,
                 alphal: ztyping.ParamTypeInput, nl: ztyping.ParamTypeInput,
                 alphar: ztyping.ParamTypeInput, nr: ztyping.ParamTypeInput,
                 obs: ztyping.ObsTypeInput, name: str = "DoubleCB", dtype: Type = ztypes.float):
        """Double sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a frac) on each side.

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
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alphal: parameter where to switch from a gaussian to the powertail on the left
            side
            nl: Exponent of the powertail on the left side
            alphar: parameter where to switch from a gaussian to the powertail on the right
            side
            nr: Exponent of the powertail on the right side
            obs:
            name:
            dtype:
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


DoubleCB.register_analytic_integral(func=double_crystalball_mu_integral, limits=crystalball_integral_limits)

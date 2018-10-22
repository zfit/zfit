from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core import tfext as ztf
from zfit.core import math as zmath
from zfit.core.basepdf import BasePDF


class Gauss(BasePDF):

    def __init__(self, mu, sigma, name="Gauss"):
        super(Gauss, self).__init__(name=name, mu=mu, sigma=sigma)

    def _unnormalized_prob(self, x):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- (x - mu) ** 2 / (ztf.constant(2.) * (sigma ** 2)))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params):
    return tf.sqrt(ztf.pi * params['sigma'])


Gauss.register_analytic_integral(func=_gauss_integral_from_inf_to_inf, dims=(0,),
                                 limits=(-zmath.inf, zmath.inf))

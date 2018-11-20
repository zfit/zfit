"""
Basic PDFs are provided here. Gauss, exponential... that can be used together with Functors to
build larger models.
"""

import math as mt

import tensorflow as tf

from zfit.core import math as zmath
from zfit.core.basepdf import BasePDF
from zfit import ztf

try:
    infinity = mt.inf
except AttributeError:  # py34
    infinity = float('inf')


class Gauss(BasePDF):

    def __init__(self, mu, sigma, name="Gauss"):  # TODO: names? TF dist?
        super(Gauss, self).__init__(name=name, mu=mu, sigma=sigma)

    def _unnormalized_pdf(self, x, norm_range=False):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- (x - mu) ** 2 / (ztf.constant(2.) * (sigma ** 2)))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params):
    # return ztf.const(1.)
    return tf.sqrt(2 * ztf.pi) * params['sigma']


Gauss.register_analytic_integral(func=_gauss_integral_from_inf_to_inf, dims=(0,),
                                 limits=(-infinity, infinity))

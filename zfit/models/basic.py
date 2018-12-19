"""
Basic PDFs are provided here. Gauss, exponential... that can be used together with Functors to
build larger models.
"""

import math as mt

import tensorflow as tf

from zfit.core import math as zmath
from zfit.core.limits import Space
from zfit.core.basepdf import BasePDF
from zfit import ztf

try:
    infinity = mt.inf
except AttributeError:  # py34
    infinity = float('inf')


class Gauss(BasePDF):

    def __init__(self, mu, sigma, obs, name="Gauss"):  # TODO: names? TF dist?
        super().__init__(name=name, obs=obs, parameters=dict(mu=mu, sigma=sigma))

    def _unnormalized_pdf(self, x, norm_range=False):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- 0.5 * tf.square((x - mu) / sigma))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params):
    # return ztf.const(1.)
    return tf.sqrt(2 * ztf.pi) * params['sigma']


# TODO: uncomment hack when switched to space
Gauss.register_analytic_integral(func=_gauss_integral_from_inf_to_inf,
                                 limits=Space.from_axes(limits=(-infinity, infinity), axes=(0,)))

from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core.basepdf import BasePDF


class Gauss(BasePDF):

    def __init__(self, mu, sigma, name="Gauss"):
        super(Gauss, self).__init__(name=name, mu=mu, sigma=sigma)

    def _func(self, value):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp((value - mu) ** 2 / (2. * sigma ** 2))
        return gauss

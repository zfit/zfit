"""
Basic PDFs are provided here. Gauss, exponential... that can be used together with Functors to
build larger models.
"""

import math as mt
from typing import Type, Any

import numpy as np
import tensorflow as tf

from zfit import ztf
from ..settings import ztypes
from ..util import ztyping
from ..core.limits import Space, ANY_LOWER, ANY_UPPER
from ..core.basepdf import BasePDF

try:
    infinity = mt.inf
except AttributeError:  # py34
    infinity = float('inf')


class CustomGaussOLD(BasePDF):

    def __init__(self, mu, sigma, obs, name="Gauss"):
        super().__init__(name=name, obs=obs, parameters=dict(mu=mu, sigma=sigma))

    def _unnormalized_pdf(self, x):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- 0.5 * tf.square((x - mu) / sigma))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params):
    # return ztf.const(1.)
    return tf.sqrt(2 * ztf.pi) * params['sigma']


CustomGaussOLD.register_analytic_integral(func=_gauss_integral_from_inf_to_inf,
                                          limits=Space.from_axes(limits=(-infinity, infinity), axes=(0,)))


class Exponential(BasePDF):

    def __init__(self, lambda_, obs: ztyping.ObsTypeInput, name: str = "Exponential",
                 **kwargs):
        """Exponential function exp(lambda * x).

        The function is normalized over a finite range and therefore a pdf. So the PDF is precisely
        defined as :math:`\\frac{ e^{\\lambda \\cdot x}}{ \\int_{lower}^{upper} e^{\\lambda \\cdot x} dx}`

        Args:
            lambda_ (zfit.Parameter): Accessed as "lambda".
            obs (Space): The Space the pdf is defined in.
            name (str): Name of the pdf.
            dtype (DType):
        """
        parameters = {'lambda': lambda_}
        super().__init__(obs, name=name, parameters=parameters, **kwargs)

    def _unnormalized_pdf(self, x):
        lambda_ = self.parameters['lambda']
        return tf.exp(lambda_ * x)


def _exp_integral_from_any_to_any(limits, params):
    lambda_ = params['lambda']

    def raw_integral(x):
        return tf.exp(lambda_ * x) / lambda_

    (lower,), (upper,) = limits.limits
    if lower[0] == - upper[0] == np.inf:
        raise NotImplementedError
    lower_int = raw_integral(x=ztf.constant(lower))
    upper_int = raw_integral(x=ztf.constant(upper))
    return upper_int - lower_int


# Exponential.register_inverse_analytic_integral()


limits = Space.from_axes(axes=0, limits=(ANY_LOWER, ANY_UPPER))
Exponential.register_analytic_integral(func=_exp_integral_from_any_to_any, limits=limits)

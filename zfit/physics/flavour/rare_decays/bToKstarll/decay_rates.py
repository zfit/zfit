from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core import interface
from . import angular_coefficients as ang


def d4Gamma(phsp, x, ml):
    """
    differential decay rate d4Gamma/dq2d3Omega of B0->K*mumu
    """

    cosThetaK = phsp.CosTheta1(x)
    cosThetaL = phsp.CosTheta2(x)
    phi = phsp.Phi(x)
    q2 = phsp.Q2(x)

    cosTheta2K = cosThetaK * cosThetaK

    sinThetaK = tf.sqrt(1.0 - cosThetaK * cosThetaK)
    sinThetaL = tf.sqrt(1.0 - cosThetaL * cosThetaL)

    sinTheta2K = (1.0 - cosThetaK * cosThetaK)
    sinTheta2L = (1.0 - cosThetaL * cosThetaL)

    sin2ThetaK = (2.0 * sinThetaK * cosThetaK)
    sin2ThetaL = (2.0 * sinThetaL * cosThetaL)

    cos2ThetaK = (2.0 * cosThetaK * cosThetaK - 1.0)  # TODO: smell, unused?
    cos2ThetaL = (2.0 * cosThetaL * cosThetaL - 1.0)

    fullPDF = ((3.0 / (8.0 * interface.Pi())) * (
                                      ang.J1s(q2, ml)   * sinTheta2K
                                      + ang.J1c(q2, ml) * cosTheta2K
                                      + ang.J2s(q2, ml) * cos2ThetaL  * sinTheta2K
                                      + ang.J2c(q2, ml) * cos2ThetaL  * cosTheta2K
                                      + ang.J3(q2, ml)  * tf.cos(2.0  * phi) * sinTheta2K * sinTheta2L
                                      + ang.J4(q2, ml)  * tf.cos(phi) * sin2ThetaK * sin2ThetaL
                                      + ang.J5(q2, ml)  * tf.cos(phi) * sin2ThetaK * sinThetaL
                                      + ang.J6s(q2, ml) * sinTheta2K  * cosThetaL
                                      + ang.J7(q2, ml)  * sin2ThetaK  * sinThetaL * tf.sin(phi)
                                      + ang.J8(q2, ml)  * sin2ThetaK  * sin2ThetaL * tf.sin(phi)
                                      + ang.J9(q2, ml)  * sinTheta2K  * sinTheta2L * tf.sin(2. * phi)))

    return fullPDF


def dGamma_dq2(q2, ml):
    """
    differential decay rate d2Gamma/dq2 of B0->K*mumu (prop to BR):
    dGamma/dq^2 = 2*J1s+J1c -1/3(2*J2s+J2c)
    """

    return 2. * ang.J1s(q2, ml) + ang.J1c(q2, ml) - 1. / 3. * (
            2. * ang.J2s(q2, ml) + ang.J2c(q2, ml))

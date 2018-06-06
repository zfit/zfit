from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core import tfext
from . import angular_coefficients as ang


def d4_gamma(phsp, x, ml):
    """
    differential decay rate d4_gamma/dq2d3Omega of B0->K*mumu
    """

    cos_theta_K = phsp.cos_theta_1(x)
    cos_theta_L = phsp.cos_theta_2(x)
    phi = phsp.phi(x)
    q2 = phsp.Q2(x)

    cos_theta2_K = cos_theta_K * cos_theta_K

    sin_theta_K = tf.sqrt(1.0 - cos_theta_K * cos_theta_K)
    sin_theta_L = tf.sqrt(1.0 - cos_theta_L * cos_theta_L)

    sin_theta2_K = (1.0 - cos_theta_K * cos_theta_K)
    sin_theta2_L = (1.0 - cos_theta_L * cos_theta_L)

    sin2_theta_K = (2.0 * sin_theta_K * cos_theta_K)
    sin2_theta_L = (2.0 * sin_theta_L * cos_theta_L)

    cos2_theta_K = (2.0 * cos_theta_K * cos_theta_K - 1.0)  # TODO: smell, unused
    cos2_theta_L = (2.0 * cos_theta_L * cos_theta_L - 1.0)

    full_pdf = ((3.0 / (8.0 * tfext.pi)) * (
        ang.J1s(q2, ml) * sin_theta2_K
        + ang.J1c(q2, ml) * cos_theta2_K
        + ang.J2s(q2, ml) * cos2_theta_L * sin_theta2_K
        + ang.J2c(q2, ml) * cos2_theta_L * cos_theta2_K
        + ang.J3(q2, ml) * tf.cos(2.0 * phi) * sin_theta2_K * sin_theta2_L
        + ang.J4(q2, ml) * tf.cos(phi) * sin2_theta_K * sin2_theta_L
        + ang.J5(q2, ml) * tf.cos(phi) * sin2_theta_K * sin_theta_L
        + ang.J6s(q2, ml) * sin_theta2_K * cos_theta_L
        + ang.J7(q2, ml) * sin2_theta_K * sin_theta_L * tf.sin(phi)
        + ang.J8(q2, ml) * sin2_theta_K * sin2_theta_L * tf.sin(phi)
        + ang.J9(q2, ml) * sin_theta2_K * sin_theta2_L * tf.sin(2. * phi)))

    return full_pdf


def d_gamma_dq2(q2, ml):
    """
    differential decay rate d2Gamma/dq2 of B0->K*mumu (prop to BR):
    dGamma/dq^2 = 2*J1s+J1c -1/3(2*J2s+J2c)
    """

    return 2. * ang.J1s(q2, ml) + ang.J1c(q2, ml) - 1. / 3. * (
        2. * ang.J2s(q2, ml) + ang.J2c(q2, ml))

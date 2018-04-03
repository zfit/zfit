from __future__ import print_function, division, absolute_import

import tensorflow as tf

from . import angular_coefficients as ang
from . import decay_rates


def SN_obs(ang_coeff):
    """Factory for S* functions.

    Args:
        ang_coeff (func(q2, ml)): Function returning the angular coefficients.
    """

    def func(q2, ml):
        return ang_coeff(q2, ml) / decay_rates.d_gamma_dq2(q2)

    return func


S1c = SN_obs(ang.J1c)
S1s = SN_obs(ang.J1s)
S2c = SN_obs(ang.J2c)
S2s = SN_obs(ang.J2s)
S3 = SN_obs(ang.J3)
S4 = SN_obs(ang.J4)
S5 = SN_obs(ang.J5)
S6s = SN_obs(ang.J6s)


def AFB(q2, ml):
    return 3. / 4. * S6s(q2, ml)


S7 = SN_obs(ang.J7)
S8 = SN_obs(ang.J8)
S9 = SN_obs(ang.J9)


# Optimized angular base

def FL(q2, ml):
    return S1c(q2, ml) - 1. / 3. * S2c(q2, ml)


def P1(q2, ml):
    return 2. * S3(q2, ml) / (1. - FL(q2, ml))


def P2(q2, ml):
    return 1. / 2. * S6s(q2, ml) / (1. - FL(q2, ml))


def P3(q2, ml):
    return -1. * S9(q2, ml) / (1. - FL(q2, ml))


def P4p(q2, ml):
    fl = FL(q2, ml)
    return S4(q2, ml) / tf.sqrt(fl * (1. - fl))


def P5p(q2, ml):
    fl = FL(q2, ml)
    return S5(q2, ml) / tf.sqrt(fl * (1. - fl))


def P6p(q2, ml):
    fl = FL(q2, ml)
    return S7(q2, ml) / tf.sqrt(fl * (1. - fl))


def P8p(q2, ml):
    fl = FL(q2, ml)
    return S8(q2, ml) / tf.sqrt(fl * (1. - fl))


def AT2(q2, ml):
    return P1(q2, ml)

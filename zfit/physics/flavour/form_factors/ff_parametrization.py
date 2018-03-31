from __future__ import print_function, division, absolute_import

import tensorflow as tf

import zfit.physics.constants as const
import zfit.physics.functions as funcs
from . import ff_parameters as ff
from . import utils


# Form factor parametrization for B0 -> K*ll

# Polynomial expansion of Form factors

def ff_poly_exp(q2, mass_res, coeff0, coeff1, coeff2):
    q2_as_z = utils.z(q2, utils.t_plus, utils.t_zero) - utils.z(0.0, utils.t_plus, utils.t_zero)
    polynom = (1. / (1. - q2 / tf.square(mass_res)) *
               (coeff0 + coeff1 * q2_as_z + coeff2 * tf.square(q2_as_z)))
    return polynom


def A0(q2):
    return ff_poly_exp(q2, utils.MR_A0, ff.A0_0, ff.A0_1, ff.A0_2)


def A1(q2):
    return ff_poly_exp(q2, utils.MR_A1, ff.A1_0, ff.A1_1, ff.A1_2)


def A12(q2):
    A12_0 = (ff.A0_0 * (tf.square(const.MB) - tf.square(const.MKst)) /
             (8.0 * const.MB * const.MKst))  # from (17) of A. Bharucha, D. Straub and R. Zwicky
    return ff_poly_exp(q2, utils.MR_A12, A12_0, ff.A12_1, ff.A12_2)


def V(q2):
    return ff_poly_exp(q2, utils.MR_V, ff.V_0, ff.V_1, ff.V_2)


def T1(q2):
    return ff_poly_exp(q2, utils.MR_T1, ff.T1_0, ff.T1_1, ff.T1_2)


def T2(q2):
    T2_0 = ff.T1_0
    return ff_poly_exp(q2, utils.MR_T2, T2_0, ff.T2_1, ff.T2_2)


def T23(q2):
    return ff_poly_exp(q2, utils.MR_T23, ff.T23_0, ff.T23_1, ff.T23_2)


# Translation between the choise of form factors as in
# C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)
# and the commonly used in the literature

def F_perp(q2):
    return tf.sqrt(2.0 * funcs.Lambda(tf.square(const.MB), tf.square(const.MKst), q2)) / (
        const.MB * (const.MB + const.MKst)) * V(q2)


def F_para(q2):
    return tf.sqrt(tf.cast(2.0, tf.float64)) * (const.MB + const.MKst) / const.MB * A1(q2)


def F_zero(q2):
    return 8.0 * const.MB * const.MKst / ((const.MB + const.MKst) * tf.sqrt(q2)) * A12(q2)


def F_time(q2):
    return A0(q2)


def F_perp_T(q2):
    return tf.sqrt(2.0 * funcs.Lambda(tf.square(const.MB), tf.square(const.MKst), q2)) / tf.square(
        const.MB) * T1(q2)


def F_para_T(q2):
    return tf.sqrt(tf.cast(2.0, tf.float64)) * (
        tf.square(const.MB) - tf.square(const.MKst)) / tf.square(const.MB) * T2(q2)


def F_zero_T(q2):
    return 4.0 * const.MKst * tf.sqrt(q2) / tf.square(const.MB + const.MKst) * T23(q2)

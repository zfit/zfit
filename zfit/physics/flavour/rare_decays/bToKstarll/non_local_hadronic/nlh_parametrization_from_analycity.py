from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.core import tfext
from zfit.core.math import poly_complex
from zfit.physics.flavour.form_factors import ff_parametrization as ff
import zfit.physics.constants as const
from zfit.physics.flavour.form_factors import utils

# Parametrization of H
from .nlh_parameters import (alpha_para_5, alpha_zero_4, alpha_zero_3, alpha_zero_2, alpha_zero_1,
                             alpha_zero_0, alpha_perp_0, alpha_perp_1, alpha_perp_2, alpha_perp_5,
                             alpha_perp_4, alpha_perp_3, alpha_para_4, alpha_para_3, alpha_para_2,
                             alpha_para_1, alpha_para_0, )

t_H_plus = 4.0 * tf.square(const.MD)
t_H_zero = 4.0 * (tf.square(const.MD) - tf.sqrt(4.0 * tf.square(const.MD)) *
                  tf.sqrt(4.0 * tf.square(const.MD) - tf.square(const.Mpsi2S)))


# function for the transformation q2->z for the parametrization of H - return Complex
#  (used only if we want to include the width of the J/psi -> q2 as complex -- not used now)
def zz(t, t_plus, t_zero):  # t is complex
    zeta = (tf.sqrt(tfext.to_complex(t_plus) - t) - tf.sqrt(
        tfext.to_complex(t_plus) - tfext.to_complex(t_zero))) / (
               tf.sqrt(tfext.to_complex(t_plus) - t) + tf.sqrt(
               tfext.to_complex(t_plus) - tfext.to_complex(t_zero)))
    return zeta


# Definition of the hadronic correlators H's
# different as in C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (arxiv:1707.07305)

def H_perp(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (tfext.to_complex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                             ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_perp_0, alpha_perp_1, alpha_perp_2,
                         alpha_perp_3, alpha_perp_4, alpha_perp_5, z_q2, real_x=True) *
            tfext.to_complex(ff.F_perp(q2)))


def H_para(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (tfext.to_complex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                             ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_para_0, alpha_para_1, alpha_para_2,
                         alpha_para_3, alpha_para_4, alpha_para_5, z_q2, real_x=True) *
            tfext.to_complex(ff.F_para(q2)))


def H_zero(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (tfext.to_complex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                             ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_zero_0, alpha_zero_1, alpha_zero_2,
                         alpha_zero_3, alpha_zero_4, z_q2, real_x=True) *
            tfext.to_complex(z_q2 - utils.z(0.0, t_H_plus, t_H_zero)) * tfext.to_complex(
            ff.F_zero(q2)))

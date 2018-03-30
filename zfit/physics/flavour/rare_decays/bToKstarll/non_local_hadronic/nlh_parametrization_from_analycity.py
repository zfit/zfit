from __future__ import print_function, division, absolute_import

from zfit.core import tfext
from zfit.core.tfext import CastComplex
from zfit.physics.flavour.form_factors import ff_parametrization as ff
import zfit.physics.constants as const
from zfit.physics.flavour.form_factors import utils
from zfit.core.interface import *

# Parametrization of H
from .nlh_parameters import (alpha_para_5, alpha_zero_4, alpha_zero_3, alpha_zero_2, alpha_zero_1,
                             alpha_zero_0, alpha_perp_0, alpha_perp_1, alpha_perp_2, alpha_perp_5,
                             alpha_perp_4, alpha_perp_3, alpha_para_4, alpha_para_3, alpha_para_2,
                             alpha_para_1, alpha_para_0, )

t_H_plus = 4.0 * tf.square(const.MD)
t_H_zero = 4.0 * tf.square(const.MD) - tf.sqrt(4.0 * tf.square(const.MD)) * tf.sqrt(
    4.0 * tf.square(const.MD) - tf.square(const.Mpsi2S))


def poly1(coeff0, coeff1, x):
    return coeff0 + coeff1 * x


def poly2(coeff0, coeff1, coeff2, x):
    return coeff0 + coeff1 * x + coeff2 * x * x  # tf.square(x)


def poly3(coeff0, coeff1, coeff2, coeff3, x):
    return coeff0 + coeff1 * x + coeff2 * x * x + coeff3 * x * x * x


def poly4(coeff0, coeff1, coeff2, coeff3, coeff4, x):
    return coeff0 + coeff1 * x + coeff2 * x * x + coeff3 * x * x * x + coeff4 * x * x * x * x  #
    # Pow of complex number doesn't work on GPU


def poly3C(coeff0, coeff1, coeff2, coeff3, x):  # take x as real and then convert as complex
    return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(
        tf.square(x)) + coeff3 * CastComplex(tf.pow(x, 3))


def poly4C(coeff0, coeff1, coeff2, coeff3, coeff4, x):  # take x as real and then convert as complex
    return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(
        tf.square(x)) + coeff3 * CastComplex(tf.pow(x, 3)) + coeff4 * CastComplex(tf.pow(x, 4))


def poly5C(coeff0, coeff1, coeff2, coeff3, coeff4, coeff5,
           x):  # take x as real and then convert as complex
    return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(
        tf.square(x)) + coeff3 * CastComplex(tf.pow(x, 3)) \
           + coeff4 * CastComplex(tf.pow(x, 4)) + coeff5 * CastComplex(tf.pow(x, 5))


def poly_complex(*args):
    """Complex polynom with the last arg being x."""
    args = list(args)
    x = CastComplex(args.pop())
    return tf.add_n([coef * x ** p for p, coef in enumerate(args)])


# function for the transformation q2->z for the parametrization of H - return Complex
#  (used only if we want to include the width of the J/psi -> q2 as complex -- not used now)
def zz(t, t_plus, t_zero):  # t is complex
    zeta = (tf.sqrt(CastComplex(t_plus) - t) - tf.sqrt(
        CastComplex(t_plus) - CastComplex(t_zero))) / (tf.sqrt(CastComplex(t_plus) - t) + tf.sqrt(
        CastComplex(t_plus) - CastComplex(t_zero)))
    return zeta


# Definition of the hadronic correlators H's
# different as in C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (arxiv:1707.07305)

def H_perp(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (CastComplex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                        ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_perp_0, alpha_perp_1, alpha_perp_2,
                         alpha_perp_3, alpha_perp_4, alpha_perp_5, z_q2) *
            CastComplex(ff.F_perp(q2)))


def H_para(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (CastComplex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                        ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_para_0, alpha_para_1, alpha_para_2,
                         alpha_para_3, alpha_para_4, alpha_para_5, z_q2) *
            CastComplex(ff.F_para(q2)))


def H_zero(q2):
    z_q2 = utils.z(q2, t_H_plus, t_H_zero)
    z_Jpsi2 = utils.z(tf.square(const.MJpsi), t_H_plus, t_H_zero)
    z_psi2S2 = utils.z(tf.square(const.Mpsi2S), t_H_plus, t_H_zero)
    return (CastComplex((1.0 - z_q2 * z_Jpsi2) * (1.0 - z_q2 * z_psi2S2) /
                        ((z_q2 - z_Jpsi2) * (z_q2 - z_psi2S2))) *
            poly_complex(alpha_zero_0, alpha_zero_1, alpha_zero_2,
                         alpha_zero_3, alpha_zero_4, z_q2) *
            CastComplex(z_q2 - utils.z(0.0, t_H_plus, t_H_zero)) * CastComplex(ff.F_zero(q2)))

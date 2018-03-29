from . import amplitudes as ampl
from zfit.physics import functions as funct
from zfit.core import optimization
from zfit.core.interface import Square, AbsSq, Real, Conj, Im

# J's definition using C. Bobeth, G. Hiller and D. van Dyk, Phys.Rev. D87 (2013) 034016


def J1s(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j1s = (2.0 + tf.square(funct.beta(q2,ml)))/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r)) \
        + (4.0 * tf.square(ml)/q2) * tf.real(A_perp_l * tf.conj(A_perp_r) + A_para_l * tf.conj(A_para_r))
    return 3./4 * j1s

def J1c(q2,ml):
    A_zero_l = ampl.A_zero_L(q2,ml)
    A_zero_r = ampl.A_zero_R(q2,ml)
    A_t      = ampl.A_time(q2)
    j1c = AbsSq(A_zero_l) + AbsSq(A_zero_r) + 4.0 * tf.square(ml)/q2 * (AbsSq(A_t) + 2.0 * tf.real(A_zero_l * tf.conj(A_zero_r)))
    return 3./4 * j1c

def J2s(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j2s = tf.square(funct.beta(q2,ml))/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r))
    return 3./4 * j2s

def J2c(q2,ml):
    A_zero_l = ampl.A_zero_L(q2,ml)
    A_zero_r = ampl.A_zero_R(q2,ml)
    j2c = - tf.square(funct.beta(q2,ml)) *(AbsSq(A_zero_l) + AbsSq(A_zero_r))
    return 3./4 * j2c

def J3(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j3 = tf.square(funct.beta(q2,ml))/2.0 * (AbsSq(A_perp_l) - AbsSq(A_para_l) + AbsSq(A_perp_r) - AbsSq(A_para_r))
    return 3./4 * j3

def J4(q2,ml):
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    A_zero_l = ampl.A_zero_L(q2,ml)
    A_zero_r = ampl.A_zero_R(q2,ml)
    j4 = tf.square(funct.beta(q2,ml))/tf.sqrt( tf.cast(2.0,tf.float64) ) * tf.real(A_zero_l * tf.conj(A_para_l) + A_zero_r * tf.conj(A_para_r))
    return 3./4 * j4

def J5(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j5 = tf.sqrt( tf.cast(2.0,tf.float64) ) * funct.beta(q2,ml) * tf.real(A_zero_l * tf.conj(A_perp_l) - A_zero_r * tf.conj(A_perp_r))
    return 3./4 * j5

def J6s(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j6s = 2.0 * funct.beta(q2,ml) * tf.real(A_para_l * tf.conj(A_perp_l) - A_para_r * tf.conj(A_perp_r))
    return 3./4 * j6s

def J7(q2,ml):
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    A_zero_l = ampl.A_zero_L(q2,ml)
    A_zero_r = ampl.A_zero_R(q2,ml)
    j7 = tf.sqrt( tf.cast(2.0,tf.float64) ) * funct.beta(q2,ml) * tf.imag(A_zero_l * tf.conj(A_para_l) - A_zero_r * tf.conj(A_para_r))
    return 3./4 * j7

def J8(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_zero_l = ampl.A_zero_L(q2,ml)
    A_zero_r = ampl.A_zero_R(q2,ml)
    j8 = tf.square(funct.beta(q2,ml)) / tf.sqrt( tf.cast(2.0,tf.float64) ) * tf.imag(A_zero_l * tf.conj(A_perp_l) + A_zero_r * tf.conj(A_perp_r))
    return 3./4 * j8

def J9(q2,ml):
    A_perp_l = ampl.A_perp_L(q2,ml)
    A_perp_r = ampl.A_perp_R(q2,ml)
    A_para_l = ampl.A_para_L(q2,ml)
    A_para_r = ampl.A_para_R(q2,ml)
    j9 = tf.square(funct.beta(q2,ml)) * tf.imag(A_perp_l * tf.conj(A_para_l) + A_perp_r * tf.conj(A_para_r))
    return 3./4 * j9


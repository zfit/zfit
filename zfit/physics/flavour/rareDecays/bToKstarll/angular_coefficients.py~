from . import amplitudes as ampl

# J's definition using C. Bobeth, G. Hiller and D. van Dyk, Phys.Rev. D87 (2013) 034016


def J1s(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j1s = (2.0 + Square(beta_l))/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r)) \
        + (4.0 * Square(ml)/q2) * Real(A_perp_l * Conj(A_perp_r) + A_para_l * Conj(A_para_r))
    return 3./4 * j1s

def J1c(q2):
    A_zero_l = ampl.A_zero_L(q2)
    A_zero_r = ampl.A_zero_R(q2)
    A_t      = ampl.A_time(q2)
    j1c = AbsSq(A_zero_l) + AbsSq(A_zero_r) + 4.0 * Square(ml)/q2 * (AbsSq(A_t) + 2.0 * Real(A_zero_l * Conj(A_zero_r))) 
    return 3./4 * j1c

def J2s(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j2s = Square(beta_l)/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r))
    return 3./4 * j2s

def J2c(q2):
    A_zero_l = ampl.A_zero_L(q2)
    A_zero_r = ampl.A_zero_R(q2)
    j2c = - Square(beta_l) *(AbsSq(A_zero_l) + AbsSq(A_zero_r))
    return 3./4 * j2c

def J3(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j3 = Square(beta_l)/2.0 * (AbsSq(A_perp_l) - AbsSq(A_para_l) + AbsSq(A_perp_r) - AbsSq(A_para_r))
    return 3./4 * j3

def J4(q2):
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    A_zero_l = ampl.A_zero_L(q2)
    A_zero_r = ampl.A_zero_R(q2)
    j4 = Square(beta_l)/Sqrt( tf.cast(2.0,tf.float64) ) * Real(A_zero_l * Conj(A_para_l) + A_zero_r * Conj(A_para_r))
    return 3./4 * j4

def J5(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j5 = Sqrt( tf.cast(2.0,tf.float64) ) * beta_l * Real(A_zero_l * Conj(A_perp_l) - A_zero_r * Conj(A_perp_r))
    return 3./4 * j5

def J6s(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j6s = 2.0 * beta_l * Real(A_para_l * Conj(A_perp_l) - A_para_r * Conj(A_perp_r))
    return 3./4 * j6s

def J7(q2):
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    A_zero_l = ampl.A_zero_L(q2)
    A_zero_r = ampl.A_zero_R(q2)
    j7 = Sqrt( tf.cast(2.0,tf.float64) ) * beta_l * Im(A_zero_l * Conj(A_para_l) - A_zero_r * Conj(A_para_r))
    return 3./4 * j7

def J8(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_zero_l = ampl.A_zero_L(q2)
    A_zero_r = ampl.A_zero_R(q2)
    j8 = Square(beta_l) / Sqrt( tf.cast(2.0,tf.float64) ) * Im(A_zero_l * Conj(A_perp_l) + A_zero_r * Conj(A_perp_r))
    return 3./4 * j8

def J9(q2):
    A_perp_l = ampl.A_perp_L(q2)
    A_perp_r = ampl.A_perp_R(q2)
    A_para_l = ampl.A_para_L(q2)
    A_para_r = ampl.A_para_R(q2)
    j9 = Square(beta_l) * Im(A_perp_l * Conj(A_para_l) + A_perp_r * Conj(A_para_r))
    return 3./4 * j9


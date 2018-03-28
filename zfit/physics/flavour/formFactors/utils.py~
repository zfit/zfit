#  from Sec.2.5 of A. Bharucha, D. Straub and R. Zwicky (BSZ2015)
t_plus = Square(MB + MKst)
t_minus = Square(MB - MKst)
t_zero = t_plus * (1. - Sqrt(1. - t_minus/t_plus))

# Mass resonances of Table 3 of BSZ2015
MR_A0  = Const(5.366)
MR_T1  = Const(5.415)
MR_V   = MR_T1
MR_T2  = Const(5.829)
MR_T23 = MR_T2
MR_A1  = MR_T2
MR_A12 = MR_T2


# function for the transformation q2->z for the parametrization of FF and real parametrization of H
def z(t, t_plus, t_zero):
  zeta = (Sqrt(t_plus - t) - Sqrt(t_plus - t_zero)) / (Sqrt(t_plus - t) + Sqrt(t_plus - t_zero))
  return zeta



def Lambda(q2):
    return tf.pow(MB,4) + tf.pow(MKst,4) + Square(q2) - 2.0 * Square(MB) * (Square(MKst) + q2) - 2.0 * q2 * Square(MKst)



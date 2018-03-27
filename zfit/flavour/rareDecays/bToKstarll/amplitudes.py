def beta(q2, ml):
    return Sqrt(1.0 - (4.*Square(ml)/q2))

# Normalization taken from C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)
def normalizeAmplitudes(q2, beta_l):
    return GF * alpha_e * Vtb * Vts * Sqrt((q2 * beta_l * Sqrt(Lambda(q2))) / (3.0 * tf.pow(tf.cast(2,tf.float64), 10) * tf.pow(Pi(),5) * MB))



# Initial implementation of the transversity amplitudes using 
# C. Bobeth, G. Hiller and D. van Dyk, Phys.Rev. D87 (2013) 034016
# Needs to be validated against C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)

def A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp):
    return CastComplex(N) * (((C9+C9p) - (C10+C10p)) * CastComplex(f_perp) \
                             + CastComplex(2.0 * (Mb + Ms) * MB / q2) * ((C7+C7p) * CastComplex(f_perp_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_perp))

def A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp):
    return CastComplex(N) * (((C9+C9p) + (C10+C10p)) * CastComplex(f_perp) \
                             + CastComplex(2.0 * (Mb + Ms) * MB / q2) * ((C7+C7p) * CastComplex(f_perp_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_perp))
    
def A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para):
    return -1.* CastComplex(N) * (((C9-C9p) - (C10-C10p)) * CastComplex(f_para) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(f_para_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_para))

def A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para):
    return -1.* CastComplex(N) * (((C9-C9p) + (C10-C10p)) * CastComplex(f_para) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(f_para_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_para))

def A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero):
    return -1.* CastComplex(N) * (((C9-C9p) - (C10-C10p)) * CastComplex(f_zero) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(f_zero_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_zero))

def A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero):
    return -1.* CastComplex(N) * (((C9-C9p) + (C10-C10p)) * CastComplex(f_zero) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(f_zero_T) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * h_zero))
  
def A_time(N, C10, C10p, q2, f_time):
    return -1.* CastComplex(N * 2.0) * (C10-C10p) * CastComplex(f_time)

  

# J's definition using C. Bobeth, G. Hiller and D. van Dyk, Phys.Rev. D87 (2013) 034016
  
def J1s(A_perp_l, A_perp_r, A_para_l, A_para_r, q2, beta_l, ml):
    j1s = (2.0 + Square(beta_l))/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r)) \
        + (4.0 * Square(ml)/q2) * Real(A_perp_l * Conj(A_perp_r) + A_para_l * Conj(A_para_r))
    return 3./4 * j1s

def J1c(A_zero_l, A_zero_r, A_t, q2, ml):
    j1c = AbsSq(A_zero_l) + AbsSq(A_zero_r) + 4.0 * Square(ml)/q2 * (AbsSq(A_t) + 2.0 * Real(A_zero_l * Conj(A_zero_r))) 
    return 3./4 * j1c

def J2s(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l):
    j2s = Square(beta_l)/4.0 * (AbsSq(A_perp_l) + AbsSq(A_para_l) + AbsSq(A_perp_r) + AbsSq(A_para_r))
    return 3./4 * j2s

def J2c(A_zero_l, A_zero_r, beta_l):
    j2c = - Square(beta_l) *(AbsSq(A_zero_l) + AbsSq(A_zero_r))
    return 3./4 * j2c

def J3(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l):
    j3 = Square(beta_l)/2.0 * (AbsSq(A_perp_l) - AbsSq(A_para_l) + AbsSq(A_perp_r) - AbsSq(A_para_r))
    return 3./4 * j3

def J4(A_para_l, A_para_r, A_zero_l, A_zero_r, beta_l):
    j4 = Square(beta_l)/Sqrt( tf.cast(2.0,tf.float64) ) * Real(A_zero_l * Conj(A_para_l) + A_zero_r * Conj(A_para_r))
    return 3./4 * j4

def J5(A_perp_l, A_perp_r, A_zero_l, A_zero_r, beta_l):
    j5 = Sqrt( tf.cast(2.0,tf.float64) ) * beta_l * Real(A_zero_l * Conj(A_perp_l) - A_zero_r * Conj(A_perp_r))
    return 3./4 * j5

def J6s(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l):
    j6s = 2.0 * beta_l * Real(A_para_l * Conj(A_perp_l) - A_para_r * Conj(A_perp_r))
    return 3./4 * j6s

def J7(A_para_l, A_para_r, A_zero_l, A_zero_r, beta_l):
    j7 = Sqrt( tf.cast(2.0,tf.float64) ) * beta_l * Im(A_zero_l * Conj(A_para_l) - A_zero_r * Conj(A_para_r))
    return 3./4 * j7

def J8(A_perp_l, A_perp_r, A_zero_l, A_zero_r, beta_l):
    j8 = Square(beta_l) / Sqrt( tf.cast(2.0,tf.float64) ) * Im(A_zero_l * Conj(A_perp_l) + A_zero_r * Conj(A_perp_r))
    return 3./4 * j8

def J9(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l):
    j9 = Square(beta_l) * Im(A_perp_l * Conj(A_para_l) + A_perp_r * Conj(A_para_r))
    return 3./4 * j9


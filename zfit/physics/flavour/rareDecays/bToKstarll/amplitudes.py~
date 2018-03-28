from zfit.zfit.flavour.form_factors import ff_parametrization as ff
from .non_local_hadronic import nlh_parametrization_from_analycity as nlh

def beta(q2, ml):
    return Sqrt(1.0 - (4.*Square(ml)/q2))

# Normalization taken from C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)
def normalizeAmplitudes(q2, beta_l):
    return GF * alpha_e * Vtb * Vts * Sqrt((q2 * beta_l * Sqrt(Lambda(q2))) / (3.0 * tf.pow(tf.cast(2,tf.float64), 10) * tf.pow(Pi(),5) * MB))



# Initial implementation of the transversity amplitudes using 
# C. Bobeth, G. Hiller and D. van Dyk, Phys.Rev. D87 (2013) 034016
# Needs to be validated against C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)

def A_perp_L(q2):
    return CastComplex(N) * (((C9+C9p) - (C10+C10p)) * CastComplex(ff.F_perp(q2)) \
                             + CastComplex(2.0 * (Mb + Ms) * MB / q2) * ((C7+C7p) * CastComplex(ff.F_perp_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_perp(q2)))

def A_perp_R(q2):
    return CastComplex(N) * (((C9+C9p) + (C10+C10p)) * CastComplex(ff.F_perp(q2)) \
                             + CastComplex(2.0 * (Mb + Ms) * MB / q2) * ((C7+C7p) * CastComplex(ff.F_perp_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_perp(q2)))
    
def A_para_L(q2):
    return -1.* CastComplex(N) * (((C9-C9p) - (C10-C10p)) * CastComplex(ff.F_para(q2)) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(ff.F_para_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_para(q2)))

def A_para_R(q2):
    return -1.* CastComplex(N) * (((C9-C9p) + (C10-C10p)) * CastComplex(ff.F_para(q2)) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(ff.F_para_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_para(q2)))

def A_zero_L(q2):
    return -1.* CastComplex(N) * (((C9-C9p) - (C10-C10p)) * CastComplex(ff.F_zero(q2)) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(ff.F_zero_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_zero(q2)))

def A_zero_R(q2):
    return -1.* CastComplex(N) * (((C9-C9p) + (C10-C10p)) * CastComplex(ff.F_zero(q2)) \
                                  + CastComplex(2.0 * (Mb - Ms) * MB / q2) * ((C7-C7p) * CastComplex(ff.F_zero_T(q2)) - CastComplex(16.0 * Square(Pi()) * MB / Mb) * nlh.H_zero(q2)))
  
def A_time(q2):
    return -1.* CastComplex(N * 2.0) * (C10-C10p) * CastComplex(f_time)

  

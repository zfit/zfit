def S1c(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    f_time   = F_time(q2)
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_t      = A_time(N, C10, C10p, q2, f_time)
    j1c = J1c(A_zero_l, A_zero_r, A_t, q2, ml)  
    return j1c / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S1s(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l)
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)  
    j1s = J1s(A_perp_l, A_perp_r, A_para_l, A_para_r, q2, beta_l, ml)
    return j1s / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S2c(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    j2c = J2c(A_zero_l, A_zero_r, beta_l)   
    return j2c / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S2s(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)  
    N = normalizeAmplitudes(q2, beta_l)
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para) 
    j2s = J2s(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l)  
    return j2s / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S3(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l) 
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)  
    j3  =  J3(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l)
    return j3 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S4(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l) 
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para) 
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)   
    j4  =  J4(A_para_l, A_para_r, A_zero_l, A_zero_r, beta_l)
    return j4 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S5(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l) 
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    j5  =  J5(A_perp_l, A_perp_r, A_zero_l, A_zero_r, beta_l)
    return j5 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S6s(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l) 
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)     
    j6s = J6s(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l)
    return j6s / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def AFB(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return 3./4. * S6s(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    
def S7(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)
    N = normalizeAmplitudes(q2, beta_l)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para) 
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)    
    j7  =  J7(A_para_l, A_para_r, A_zero_l, A_zero_r, beta_l)
    return j7 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S8(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml) 
    N = normalizeAmplitudes(q2, beta_l)
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_zero   = F_zero(q2)
    f_zero_T = F_zero_T(q2)
    h_zero   = H_zero(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_zero)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_zero_l = A_zero_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)
    A_zero_r = A_zero_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_zero, f_zero_T, h_zero)  
    j8  =  J8(A_perp_l, A_perp_r, A_zero_l, A_zero_r, beta_l)
    return j8 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

def S9(q2, C7, C7p, C9, C9p, C10, C10p, ml ): 
    beta_l = beta(q2, ml)   
    N = normalizeAmplitudes(q2, beta_l)
    f_perp   = F_perp(q2)
    f_perp_T = F_perp_T(q2)
    h_perp   = H_perp(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_perp)
    f_para   = F_para(q2)
    f_para_T = F_para_T(q2) 
    h_para   = H_para(z(q2, t_H_plus, t_H_zero)) * CastComplex(f_para)
    A_perp_l = A_perp_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_perp_r = A_perp_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_perp, f_perp_T, h_perp)
    A_para_l = A_para_L(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)
    A_para_r = A_para_R(N, C7, C7p, C9, C9p, C10, C10p, q2, f_para, f_para_T, h_para)   
    j9  =  J9(A_perp_l, A_perp_r, A_para_l, A_para_r, beta_l)
    return j9 / dGamma_dq2(q2, C7, C7p, C9, C9p, C10, C10p, ml )  

# Optimized angular base

def FL(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return S1c(q2, C7, C7p, C9, C9p, C10, C10p, ml) - 1./3. * S2c(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    
def P1(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return 2. * S3(q2, C7, C7p, C9, C9p, C10, C10p, ml) / (1. - FL(q2, C7, C7p, C9, C9p, C10, C10p, ml) )

def P2(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return 1./2. * S6s(q2, C7, C7p, C9, C9p, C10, C10p, ml) / (1. - FL(q2, C7, C7p, C9, C9p, C10, C10p, ml) )

def P3(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return -1. * S9(q2, C7, C7p, C9, C9p, C10, C10p, ml) / (1. - FL(q2, C7, C7p, C9, C9p, C10, C10p, ml) )

def P4p(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    fl = FL(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    return S4(q2, C7, C7p, C9, C9p, C10, C10p, ml) / Sqrt( fl * (1. - fl) )

def P5p(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    fl = FL(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    return S5(q2, C7, C7p, C9, C9p, C10, C10p, ml) / Sqrt( fl * (1. - fl) )

def P6p(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    fl = FL(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    return S7(q2, C7, C7p, C9, C9p, C10, C10p, ml) / Sqrt( fl * (1. - fl) )

def P8p(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    fl = FL(q2, C7, C7p, C9, C9p, C10, C10p, ml)
    return S8(q2, C7, C7p, C9, C9p, C10, C10p, ml) / Sqrt( fl * (1. - fl) )

def AT2(q2, C7, C7p, C9, C9p, C10, C10p, ml ):    
    return P1(q2, C7, C7p, C9, C9p, C10, C10p, ml)



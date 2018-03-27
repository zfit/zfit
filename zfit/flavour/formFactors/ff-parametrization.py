# Form factor parametrization for B0 -> K*ll

# Polynomial expansion of Form factors
def A0(q2):
    return  1. / (1. - q2/Square(MR_A0) ) * ( A0_0  + A0_1  * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + A0_2  * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def A1(q2):
    return 1. / (1. - q2/Square(MR_A1) ) * ( A1_0  + A1_1  * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + A1_2  * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def A12(q2):
    A12_0 =  A0_0 * (Square(MB) - Square(MKst)) / (8.0 * MB * MKst) # from (17) of A. Bharucha, D. Straub and R. Zwicky
    return 1. / (1. - q2/Square(MR_A12)) * ( A12_0 + A12_1 * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + A12_2 * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def V(q2):
    return 1. / (1. - q2/Square(MR_V)  ) * ( V_0   + V_1   * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + V_2   * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def T1(q2):
    return 1. / (1. - q2/Square(MR_T1 )) * ( T1_0  + T1_1  * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + T1_2  * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def T2(q2):
    T2_0  =  T1_0 
    return 1. / (1. - q2/Square(MR_T2) ) * ( T2_0  + T2_1  * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + T2_2  * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )

def T23(q2):
    return 1. / (1. - q2/Square(MR_T23)) * ( T23_0 + T23_1 * (z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) + T23_2 * Square(z(q2,t_plus,t_zero) - z(0.0,t_plus,t_zero)) )


# Translation between the choise of form factors as in 
# C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (in preparation)
# and the commonly used in the literature

def F_perp(q2):
    return Sqrt(2.0*Lambda(q2)) / (MB * (MB + MKst)) * V(q2) 
  
def  F_para(q2):
    return Sqrt( tf.cast(2.0,tf.float64) ) * (MB + MKst) / MB * A1(q2) 

def  F_zero(q2):
    return 8.0 * MB * MKst / ((MB + MKst) * Sqrt(q2)) * A12(q2) 

def  F_time(q2): 
    return A0(q2) 

def  F_perp_T(q2):  
    return Sqrt(2.0*Lambda(q2)) / Square(MB) * T1(q2) 

def  F_para_T(q2):
    return Sqrt( tf.cast(2.0,tf.float64) ) * (Square(MB) - Square(MKst)) / Square(MB) * T2(q2) 

def  F_zero_T(q2):
    return 4.0 * MKst * Sqrt(q2) / Square(MB + MKst) * T23(q2)   



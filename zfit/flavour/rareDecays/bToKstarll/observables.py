from . import angular_coefficients as ang
from . import decay_rates

def S1c(q2): 
    return ang.J1c(q2) / decay_rates.dGamma_dq2(q2)  

def S1s(q2): 
    return ang.J1s(q2) / decay_rates.dGamma_dq2(q2)  

def S2c(q2): 
    return ang.J2c(q2) / decay_rates.dGamma_dq2(q2)  

def S2s(q2): 
    return ang.J2s(q2) / decay_rates.dGamma_dq2(q2)  

def S3(q2): 
    return ang.J3(q2) / decay_rates.dGamma_dq2(q2)  

def S4(q2): 
    return ang.J4(q2) / decay_rates.dGamma_dq2(q2)  

def S5(q2): 
    return ang.J5(q2) / decay_rates.dGamma_dq2(q2)  

def S6s(q2): 
    return ang.J6s(q2) / decay_rates.dGamma_dq2(q2)  

def AFB(q2):    
    return 3./4. * S6s(q2)
    
def S7(q2): 
    return ang.J7(q2) / decay_rates.dGamma_dq2(q2)  

def S8(q2): 
    return ang.J8(q2) / decay_rates.dGamma_dq2(q2)  

def S9(q2): 
    return ang.J9(q2) / decay_rates.dGamma_dq2(q2)  




# Optimized angular base

def FL(q2):    
    return S1c(q2) - 1./3. * S2c(q2)
    
def P1(q2):    
    return 2. * S3(q2) / (1. - FL(q2) )

def P2(q2):    
    return 1./2. * S6s(q2) / (1. - FL(q2) )

def P3(q2):    
    return -1. * S9(q2) / (1. - FL(q2) )

def P4p(q2):    
    fl = FL(q2)
    return S4(q2) / Sqrt( fl * (1. - fl) )

def P5p(q2):    
    fl = FL(q2)
    return S5(q2) / Sqrt( fl * (1. - fl) )

def P6p(q2):    
    fl = FL(q2)
    return S7(q2) / Sqrt( fl * (1. - fl) )

def P8p(q2):    
    fl = FL(q2)
    return S8(q2) / Sqrt( fl * (1. - fl) )

def AT2(q2):    
    return P1(q2)



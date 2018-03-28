from . import angular_coefficients as ang
from . import decay_rates

def S1c(q2,ml): 
    return ang.J1c(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S1s(q2,ml): 
    return ang.J1s(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S2c(q2,ml): 
    return ang.J2c(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S2s(q2,ml): 
    return ang.J2s(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S3(q2,ml): 
    return ang.J3(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S4(q2,ml): 
    return ang.J4(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S5(q2,ml): 
    return ang.J5(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S6s(q2,ml): 
    return ang.J6s(q2,ml) / decay_rates.dGamma_dq2(q2)  

def AFB(q2,ml):    
    return 3./4. * S6s(q2,ml)
    
def S7(q2,ml): 
    return ang.J7(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S8(q2,ml): 
    return ang.J8(q2,ml) / decay_rates.dGamma_dq2(q2)  

def S9(q2,ml): 
    return ang.J9(q2,ml) / decay_rates.dGamma_dq2(q2)  




# Optimized angular base

def FL(q2,ml):    
    return S1c(q2,ml) - 1./3. * S2c(q2,ml)
    
def P1(q2,ml):    
    return 2. * S3(q2,ml) / (1. - FL(q2,ml) )

def P2(q2,ml):    
    return 1./2. * S6s(q2,ml) / (1. - FL(q2,ml) )

def P3(q2,ml):    
    return -1. * S9(q2,ml) / (1. - FL(q2,ml) )

def P4p(q2,ml):    
    fl = FL(q2,ml)
    return S4(q2,ml) / Sqrt( fl * (1. - fl) )

def P5p(q2,ml):    
    fl = FL(q2,ml)
    return S5(q2,ml) / Sqrt( fl * (1. - fl) )

def P6p(q2,ml):    
    fl = FL(q2,ml)
    return S7(q2,ml) / Sqrt( fl * (1. - fl) )

def P8p(q2,ml):    
    fl = FL(q2,ml)
    return S8(q2,ml) / Sqrt( fl * (1. - fl) )

def AT2(q2,ml):    
    return P1(q2,ml)



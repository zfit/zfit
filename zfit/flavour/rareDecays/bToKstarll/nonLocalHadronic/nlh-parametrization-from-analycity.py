from zfit.zfit.flavour.form_factors import ff_parametrization as ff

# Parametrization of H

t_H_plus = 4.0 * Square(MD)
t_H_zero = 4.0 * Square(MD) - Sqrt(4.0 * Square(MD)) * Sqrt(4.0 * Square(MD) - Square(Mpsi2S))



def poly1(coeff0, coeff1, x):
  return coeff0 + coeff1 * x 

def poly2(coeff0, coeff1, coeff2, x):
  return coeff0 + coeff1 * x + coeff2 * x * x # Square(x)

def poly3(coeff0, coeff1, coeff2, coeff3, x):
  return coeff0 + coeff1 * x + coeff2 * x * x + coeff3 * x * x * x

def poly4(coeff0, coeff1, coeff2, coeff3, coeff4, x):
  return coeff0 + coeff1 * x + coeff2 * x * x + coeff3 * x * x * x + coeff4 * x * x * x * x  # Pow of complex number doesn't work on GPU


def poly3C(coeff0, coeff1, coeff2, coeff3, x): # take x as real and then convert as complex
  return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(Square(x)) + coeff3 * CastComplex(Pow(x,3))

def poly4C(coeff0, coeff1, coeff2, coeff3, coeff4, x): # take x as real and then convert as complex
  return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(Square(x)) + coeff3 * CastComplex(Pow(x,3)) + coeff4 * CastComplex(Pow(x,4))

def poly5C(coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, x): # take x as real and then convert as complex
  return coeff0 + coeff1 * CastComplex(x) + coeff2 * CastComplex(Square(x)) + coeff3 * CastComplex(Pow(x,3))  \
         + coeff4 * CastComplex(Pow(x,4)) + coeff5 * CastComplex(Pow(x,5)) 


# function for the transformation q2->z for the parametrization of H - return Complex 
#  (used only if we want to include the width of the J/psi -> q2 as complex -- not used now)
def zz(t, t_plus, t_zero): # t is complex
  zeta = (Sqrt(CastComplex(t_plus) - t) - Sqrt(CastComplex(t_plus) - CastComplex(t_zero))) / (Sqrt(CastComplex(t_plus) - t) + Sqrt(CastComplex(t_plus) - CastComplex(t_zero)))
  return zeta



# Definition of the hadronic correlators H's
# different as in C. Bobeth, M. Chrzaszcz, D. van Dyk and J. Virto (arxiv:1707.07305)

def H_perp(q2):
  z_q2 = z(q2, t_H_plus, t_H_zero)
  return CastComplex((1.0 - z_q2 * z(Square(MJpsi),t_H_plus,t_H_zero)) * (1.0 - z_q2 * z(Square(Mpsi2S),t_H_plus,t_H_zero)) \
         / ((z_q2 - z(Square(MJpsi),t_H_plus,t_H_zero)) * (z_q2 - z(Square(Mpsi2S),t_H_plus,t_H_zero)))) \
         * poly5C(alpha_perp_0, alpha_perp_1, alpha_perp_2, alpha_perp_3, alpha_perp_4, alpha_perp_5, z_q2) * CastComplex(ff.F_perp(q2))

def H_para(q2):
  z_q2 = z(q2, t_H_plus, t_H_zero)
  return CastComplex((1.0 - z_q2 * z(Square(MJpsi),t_H_plus,t_H_zero)) * (1.0 - z_q2 * z(Square(Mpsi2S),t_H_plus,t_H_zero)) \
         / ((z_q2 - z(Square(MJpsi),t_H_plus,t_H_zero)) * (z_q2 - z(Square(Mpsi2S),t_H_plus,t_H_zero)))) \
         * poly5C(alpha_para_0, alpha_para_1, alpha_para_2, alpha_para_3, alpha_para_4, alpha_para_5, z_q2) * CastComplex(ff.F_para(q2))


def H_zero(q2):
  z_q2 = z(q2, t_H_plus, t_H_zero)
  return CastComplex((1.0 - z_q2 * z(Square(MJpsi),t_H_plus,t_H_zero)) * (1.0 - z_q2 * z(Square(Mpsi2S),t_H_plus,t_H_zero)) \
           / ((z_q2 - z(Square(MJpsi),t_H_plus,t_H_zero)) * (z_q2 - z(Square(Mpsi2S),t_H_plus,t_H_zero)))) \
           * poly4C(alpha_zero_0, alpha_zero_1, alpha_zero_2, alpha_zero_3, alpha_zero_4, z_q2) \
           * CastComplex(z_q2 - z(0.0,t_H_plus,t_H_zero)) * CastComplex(ff.F_zero(q2))


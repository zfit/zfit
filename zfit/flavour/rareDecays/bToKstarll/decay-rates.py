from . import angular_coefficients as ang



def d4Gamma(phsp, x):
  """
  differential decay rate d4Gamma/dq2d3Omega of B0->K*mumu
  """

  cosThetaK = phsp.CosTheta1(x)
  cosThetaL = phsp.CosTheta2(x)
  phi       = phsp.Phi(x)
  q2        = phsp.Q2(x)
  
  cosTheta2K = cosThetaK * cosThetaK
    
  sinThetaK = Sqrt( 1.0 - cosThetaK * cosThetaK )
  sinThetaL = Sqrt( 1.0 - cosThetaL * cosThetaL )

  sinTheta2K =  (1.0 - cosThetaK * cosThetaK)
  sinTheta2L =  (1.0 - cosThetaL * cosThetaL)

  sin2ThetaK = (2.0 * sinThetaK * cosThetaK)
  sin2ThetaL = (2.0 * sinThetaL * cosThetaL)

  cos2ThetaK = (2.0 * cosThetaK * cosThetaK - 1.0)
  cos2ThetaL = (2.0 * cosThetaL * cosThetaL - 1.0)

  
  fullPDF = (3.0 /(8.0 *Pi())) * (  \
                                  ang.J1s(q2) * sinTheta2K    \
                                + ang.J1c(q2) * cosTheta2K    \
                                + ang.J2s(q2) * cos2ThetaL * sinTheta2K    \
                                + ang.J2c(q2) * cos2ThetaL * cosTheta2K    \
                                + ang.J3(q2)  * Cos(2.0 * phi) * sinTheta2K * sinTheta2L   \
                                + ang.J4(q2)  * Cos(phi) * sin2ThetaK * sin2ThetaL      \
                                + ang.J5(q2)  * Cos(phi) * sin2ThetaK * sinThetaL       \
                                + ang.J6s(q2) * sinTheta2K * cosThetaL                  \
                                + ang.J7(q2)  * sin2ThetaK * sinThetaL * Sin(phi)       \
                                + ang.J8(q2)  * sin2ThetaK * sin2ThetaL * Sin(phi)      \
                                + ang.J9(q2)  * sinTheta2K * sinTheta2L * Sin(2.*phi) )
    
  return fullPDF




def dGamma_dq2(q2): 
  """
  differential decay rate d2Gamma/dq2 of B0->K*mumu (prop to BR):  
  dGamma/dq^2 = 2*J1s+J1c -1/3(2*J2s+J2c)
  """

  return 2. * ang.J1s(q2) + ang.J1c(q2) - 1./3. * (2.*ang.J2s(q2) + ang.J2c(q2))


from Interface import *
from Kinematics import *

def HelicityAmplitude(x, spin) : 
  """
  Helicity amplitude for a resonance in scalar-scalar state
    x    : cos(helicity angle)
    spin : spin of the resonance
  """
  if spin == 0 : return Complex( Const(1.), Const(0.))
  if spin == 1 : return Complex( x, Const(0.))
  if spin == 2 : return Complex( (3.*x**2-1.)/2., Const(0.))
  if spin == 3 : return Complex( (5.*x**3-3.*x)/2., Const(0.))
  if spin == 4 : return Complex( (35.*x**4-30.*x**2+3.)/8., Const(0.))
  return None

def RelativisticBreitWigner(m2, mres, wres) : 
  """
  Relativistic Breit-Wigner 
  """
  if wres.dtype is ctype : 
    return 1./(CastComplex(mres**2 - m2) - Complex(Const(0.), mres)*wres)
  if wres.dtype is fptype : 
    return 1./Complex(mres**2 - m2, -mres*wres)
  return None

def BlattWeisskopfFormFactor(q, q0, d, l) : 
  """
  Blatt-Weisskopf formfactor for intermediate resonance
  """
  z  = q*d
  z0 = q0*d
  def hankel1(x):
    if l == 0 : return Const(1.)
    if l == 1 : return 1 + x**2
    if l == 2 : 
      x2 = x**2
      return 9 + x2*(3. + x2)
    if l == 3 : 
      x2 = x**2
      return 225 + x2*(45 + x2*(6 + x2))
    if l == 4 : 
      x2 = x**2
      return 11025. + x2*(1575. + x2*(135. + x2*(10. + x2)))
  return Sqrt(hankel1(z0)/hankel1(z))

def MassDependentWidth(m, m0, gamma0, p, p0, ff, l) : 
  """
  Mass-dependent width for BW amplitude
  """
  return gamma0*((p/p0)**(2*l+1))*(m0/m)*(ff**2)

def OrbitalBarrierFactor(p, p0, l) : 
  """
  Orbital barrier factor
  """
  return (p/p0)**l

def BreitWignerLineShape(m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrierFactor = True, ma0=None, md0 = None) : 
  """
  Breit-Wigner amplitude with Blatt-Weisskopf formfactors, mass-dependent width and orbital barriers
  """
  m = Sqrt(m2)
  q  = TwoBodyMomentum(md, m, mc)
  q0 = TwoBodyMomentum(md if md0 is None else md0, m0, mc)
  p  = TwoBodyMomentum(m, ma, mb)
  p0 = TwoBodyMomentum(m0, ma if ma0 is None else ma0, mb)
  ffr = BlattWeisskopfFormFactor(p, p0, dr, lr)
  ffd = BlattWeisskopfFormFactor(q, q0, dd, ld)
  width = MassDependentWidth(m, m0, gamma0, p, p0, ffr, lr)
  bw = RelativisticBreitWigner(m2, m0, width)
  ff = ffr*ffd
  if barrierFactor : 
    b1 = OrbitalBarrierFactor(p, p0, lr)
    b2 = OrbitalBarrierFactor(q, q0, ld)
    ff *= b1*b2
  return bw*Complex(ff, Const(0.))

def SubThresholdBreitWignerLineShape(m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrierFactor = True) : 
  """
  Breit-Wigner amplitude (with the mass under kinematic threshold) 
  with Blatt-Weisskopf formfactors, mass-dependent width and orbital barriers
  """
  m = Sqrt(m2)
  mmin = ma+mb
  mmax = md-mc
  tanhterm = Tanh( (m0 - ( (mmin+mmax)/2.) )/(mmax-mmin) )
  m0eff = mmin + (mmax-mmin)*(1.+tanhterm)/2.
  q  = TwoBodyMomentum(md, m, mc)
  q0 = TwoBodyMomentum(md, m0eff, mc)
  p  = TwoBodyMomentum(m, ma, mb)
  p0 = TwoBodyMomentum(m0eff, ma, mb)
  ffr = BlattWeisskopfFormFactor(p, p0, dr, lr)
  ffd = BlattWeisskopfFormFactor(q, q0, dd, ld)
  width = MassDependentWidth(m, m0, gamma0, p, p0, ffr, lr)
  bw = RelativisticBreitWigner(m2, m0, width)
  ff = ffr*ffd
  if barrierFactor : 
    b1 = OrbitalBarrierFactor(p, p0, lr)
    b2 = OrbitalBarrierFactor(q, q0, ld)
    ff *= b1*b2
  return bw*Complex(ff, Const(0.))

def ExponentialNonResonantLineShape(m2, m0, alpha, ma, mb, mc, md, lr, ld, barrierFactor = True) : 
  """
  Exponential nonresonant amplitude with orbital barriers
  """
  if barrierFactor : 
    m = Sqrt(m2)
    q  = TwoBodyMomentum(md, m, mc)
    q0 = TwoBodyMomentum(md, m0, mc)
    p  = TwoBodyMomentum(m, ma, mb)
    p0 = TwoBodyMomentum(m0, ma, mb)
    b1 = OrbitalBarrierFactor(p, p0, lr)
    b2 = OrbitalBarrierFactor(q, q0, ld)
    return Complex(b1*b2*Exp(-alpha*(m2-m0**2)), Const(0.))
  else : 
    return Complex(Exp(-alpha*(m2-m0**2)), Const(0.))

def GounarisSakuraiLineShape(s, m, gamma, m_pi) : 
  """
    Gounaris-Sakurai shape for rho->pipi
      s     : squared pipi inv. mass
      m     : rho mass
      gamma : rho width
      m_pi  : pion mass
  """
  m2 = m*m
  m_pi2 = m_pi*m_pi
  ss = Sqrt(s)

  ppi2 = (s-4.*m_pi2)/4.
  p02 = (m2-4.*m_pi2)/4.
  p0 = Sqrt(p02)
  ppi = Sqrt(ppi2)

  hs = 2.*ppi/Pi()/ss*Log((ss+2.*ppi)/2./m_pi)
  hm = 2.*p0/Pi()/m*Log((m+2.*ppi)/2./m_pi)

  dhdq = hm*(1./8./p02 - 1./2./m2) + 1./2./Pi()/m2
  f = gamma*m2/(p0**3)*(ppi2*(hs-hm) - p02*(s-m2)*dhdq)

  gamma_s = gamma*m2*(ppi**3)/s/(p0**3)

  dr = m2-s+f
  di = ss*gamma_s

  r = dr/(dr**2+di**2)
  i = di/(dr**2+di**2)

  return Complex(r,i)

def FlatteLineShape(s, m, g1, g2, ma1, mb1, ma2, mb2) : 
  """
    Flatte line shape
      s : squared inv. mass
      m : resonance mass
      g1 : coupling to ma1, mb1
      g2 : coupling to ma2, mb2
  """
  mab = Sqrt(s)
  pab1 = TwoBodyMomentum(mab, ma1, mb1)
  rho1 = 2.*pab1/mab
  pab2 = ComplexTwoBodyMomentum(mab, ma2, mb2)
  rho2 = 2.*pab2/CastComplex(mab)
  gamma = (CastComplex(g1**2*rho1) + CastComplex(g2**2)*rho2)/CastComplex(m)
  return RelativisticBreitWigner(s, m, gamma)

# New! Follow the definition of (not completely)LHCb-PAPER-2016-141, and Rafa's thesis(Laura++)
def LASSLineShape(m2ab, m0, gamma0, a, r, ma, mb, dr, lr, barrierFactor = True) : 
  """
  LASS amplitude with Blatt-Weisskopf formfactors, mass-dependent width and orbital barriers
  """
  m  = Sqrt(m2ab)
  q  = TwoBodyMomentum(m, ma, mb)
  q0 = TwoBodyMomentum(m0, ma, mb)
  ffr = BlattWeisskopfFormFactor(q, q0, dr, lr)
  nonResLASS = NonresonantLASSLineShape(m2ab, a, r, ma, mb)
  resLASS    = ResonantLASSLineShape(m2ab, m0, gamma0, a, r, ma, mb)
  ff = ffr
  if barrierFactor : 
    b1 = OrbitalBarrierFactor(q, q0, lr)
    ff *= b1
  return Complex(ff, Const(0.))*(nonResLASS+resLASS)

def NonresonantLASSLineShape(m2ab, a, r, ma, mb) : 
  """
    LASS line shape, nonresonant part
  """
  m = Sqrt(m2ab)
  q = TwoBodyMomentum(m, ma, mb)
  cot_deltab = 1./a/q + 1./2.*r*q
  ampl = CastComplex(m)/Complex(q*cot_deltab, -q)
  return ampl

# Modified to match the definition of LHCb-PAPER-2016-141 (and also Laura++)
def ResonantLASSLineShape(m2ab, m0, gamma0, a, r, ma, mb) : 
  """
    LASS line shape, resonant part
  """
  m = Sqrt(m2ab)
  q0 = TwoBodyMomentum(m0, ma, mb)
  q = TwoBodyMomentum(m, ma, mb)
  cot_deltab = 1./a/q + 1./2.*r*q
  phase = Atan(1./cot_deltab)
  width = gamma0*q/m*m0/q0
  ampl = RelativisticBreitWigner(m2ab, m0, width)*Complex(Cos(2.*phase), Sin(2.*phase))*CastComplex(m0*m0*gamma0/q0) 
  return ampl

def DabbaLineShape(m2ab, b, alpha, beta, ma, mb) :
  """
    Dabba line shape
  """
  mSum = ma + mb
  m2a = ma**2
  m2b = mb**2
  sAdler = max(m2a, m2b) - 0.5*min(m2a, m2b)
  mSum2 = mSum*mSum
  mDiff = m2ab - mSum2
  rho = Sqrt(1. - mSum2/m2ab)
  realPart = 1.0 - beta*mDiff
  imagPart = b*Exp(-alpha*mDiff)*(m2ab-sAdler)*rho
  denomFactor = realPart*realPart + imagPart*imagPart
  ampl = Complex(realPart, imagPart)/CastComplex(denomFactor)
  return ampl

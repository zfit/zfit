import sys
import operator
import tensorflow as tf
import numpy as np
import math
import itertools
import Optimisation

from Interface import *

def SpatialComponents(vector) :
  """
  Return spatial components of the input Lorentz vector
    vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
  """
#  return tf.slice(vector, [0, 0], [-1, 3])
  return vector[:,0:3]

def TimeComponent(vector) :
  """
  Return time component of the input Lorentz vector
    vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
  """
#  return tf.unstack(vector, axis=1)[3]
  return vector[:,3]

def XComponent(vector) :
  """
  Return spatial X component of the input Lorentz or 3-vector
    vector : input vector
  """
#  return tf.unstack(vector, axis=1)[0]
  return vector[:,0]

def YComponent(vector) :
  """
  Return spatial Y component of the input Lorentz or 3-vector
    vector : input vector
  """
  return vector[:,1]

def ZComponent(vector) :
  """
  Return spatial Z component of the input Lorentz or 3-vector
    vector : input vector
  """
  return vector[:,2]

def Vector(x, y, z) :
  """
  Make a 3-vector from components
    x, y, z : vector components
  """
  return tf.stack( [x, y, z], axis=1 )

def Scalar(x) :
  """
  Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
  One cannot do e.g. Const(2.)*Vector(x, y, z), needs to do Scalar(Const(2))*Vector(x, y, z)
  """
  return tf.stack( [x], axis=1 )

def LorentzVector(space, time) :
  """
  Make a Lorentz vector from spatial and time components
    space : 3-vector of spatial components
    time  : time component
  """
  return tf.concat([ space, tf.stack( [ time ], axis=1) ], axis = 1 )

def MetricTensor() : 
  """
  Metric tensor for Lorentz space (constant)
  """
  return tf.constant( [ -1., -1., -1., 1. ], dtype=fptype )

def Mass(vector) :
  """
  Calculate mass scalar for Lorentz 4-momentum
    vector : input Lorentz momentum vector
  """
  return Sqrt(tf.reduce_sum( vector*vector*MetricTensor(), 1 ))

def ScalarProduct(vec1, vec2) :
  """
  Calculate scalar product of two 3-vectors
  """
  return tf.reduce_sum(vec1*vec2, 1)

def VectorProduct(vec1, vec2) :
  """
  Calculate vector product of two 3-vectors
  """
  return tf.cross(vec1, vec2)

def CrossProduct(vec1, vec2) :
  """
  Calculate cross product of two 3-vectors
  """
  return tf.cross(vec1,vec2)

def Norm(vec) :
  """
  Calculate norm of 3-vector
  """
  return Sqrt( tf.reduce_sum(vec*vec, 1) )

def UnitVector(vec) :
  """
    Return unit vector in the direction of vec
  """
  return vec/Scalar(Norm(vec))

def PerpendicularUnitVector(vec1, vec2) :
  """
    Return unit vector perpendicular to the plane formed by vec1 and vec2
  """
  v = VectorProduct(vec1, vec2)
  return v/Scalar(Norm(v))

def LorentzBoost(vector, boostvector) :
  """
  Perform Lorentz boost
    vector :     4-vector to be boosted
    boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components are used)
  """
  boost = SpatialComponents(boostvector)
  b2 = ScalarProduct(boost, boost)
  gamma = 1./Sqrt(1.-b2)
  gamma2 = (gamma-1.0)/b2
  ve = TimeComponent(vector)
  vp = SpatialComponents(vector)
  bp = ScalarProduct(vp, boost)
  vp2 = vp + Scalar(gamma2*bp + gamma*ve)*boost
  ve2 = gamma*(ve + bp)
  return LorentzVector(vp2, ve2)

def BoostToRest(vector, boostvector) :
  """
    Perform Lorentz boost to the rest frame of the 4-vector boostvector.
  """
  boost = -SpatialComponents(boostvector)/Scalar(TimeComponent(boostvector))
  return LorentzBoost(vector, boost)

def BoostFromRest(vector, boostvector) : 
  """
    Perform Lorentz boost from the rest frame of the 4-vector boostvector.
  """
  boost = SpatialComponents(boostvector)/TimeComponent(boostvector)
  return LorentzBoost(vector, boost)

def RotateVector(v, phi, theta, psi) :
  """
  Perform 3D rotation of the 3-vector
    v : vector to be rotated
    phi, theta, psi : Euler angles in Z-Y-Z convention
  """

  # Rotate Z (phi)
  c1 = Cos(phi)
  s1 = Sin(phi)
  c2 = Cos(theta)
  s2 = Sin(theta)
  c3 = Cos(psi)
  s3 = Sin(psi)

  # Rotate Y (theta)
  fzx2 =-s2*c1
  fzy2 = s2*s1
  fzz2 = c2

  # Rotate Z (psi)
  fxx3 = c3*c2*c1 - s3*s1
  fxy3 =-c3*c2*s1 - s3*c1
  fxz3 = c3*s2
  fyx3 = s3*c2*c1 + c3*s1
  fyy3 =-s3*c2*s1 + c3*c1
  fyz3 = s3*s2

  # Transform v
  vx = XComponent(v)
  vy = YComponent(v)
  vz = ZComponent(v)

  _vx = fxx3*vx + fxy3*vy + fxz3*vz
  _vy = fyx3*vx + fyy3*vy + fyz3*vz
  _vz = fzx2*vx + fzy2*vy + fzz2*vz

  return Vector(_vx, _vy, _vz)

def RotateLorentzVector(v, phi, theta, psi):
  """
  Perform 3D rotation of the 4-vector
    v : vector to be rotated
    phi, theta, psi : Euler angles in Z-Y-Z convention
  """
  return LorentzVector(RotateVector(SpatialComponents(v), phi, theta, psi), TimeComponent(v))

def ProjectLorentzVector(p, (x1, y1, z1)):
  p0 = SpatialComponents(p)
  p1 = LorentzVector( Vector( ScalarProduct(x1, p0), ScalarProduct(y1, p0), ScalarProduct(z1, p0) ), TimeComponent(p) )
  return p1

def CosHelicityAngleDalitz(m2ab, m2bc, md, ma, mb, mc) :
  """
  Calculate cos(helicity angle) for set of two Dalitz plot variables
    m2ab, m2bc : Dalitz plot variables (inv. masses squared of AB and BC combinations)
    md : mass of the decaying particle
    ma, mb, mc : masses of final state particles
  """
  md2 = md**2
  ma2 = ma**2
  mb2 = mb**2
  mc2 = mc**2
  m2ac = md2 + ma2 + mb2 + mc2 - m2ab - m2bc
  mab = Sqrt(m2ab)
  mac = Sqrt(m2ac)
  mbc = Sqrt(m2bc)
  p2a = 0.25/md2*(md2-(mbc+ma)**2)*(md2-(mbc-ma)**2)
  p2b = 0.25/md2*(md2-(mac+mb)**2)*(md2-(mac-mb)**2)
  p2c = 0.25/md2*(md2-(mab+mc)**2)*(md2-(mab-mc)**2)
  eb = (m2ab-ma2+mb2)/2./mab
  ec = (md2-m2ab-mc2)/2./mab
  pb = Sqrt(eb**2-mb2)
  pc = Sqrt(ec**2-mc2)
  e2sum = (eb+ec)**2
  m2bc_max = e2sum-(pb-pc)**2
  m2bc_min = e2sum-(pb+pc)**2
  return (m2bc_max + m2bc_min - 2.*m2bc)/(m2bc_max-m2bc_min)

def SphericalAngles(pb) : 
  """
    theta, phi : polar and azimuthal angles of the vector pb
  """
  z1 = UnitVector(SpatialComponents(pb))       # New z-axis is in the direction of pb
  theta = Acos(ZComponent(z1))                 # Helicity angle
  phi = Atan2(YComponent(pb), XComponent(pb))  # Phi angle
  return (theta, phi)

def HelicityAngles(pb) :
  """
    theta, phi : polar and azimuthal angles of the vector pb
  """
  return SphericalAngles(pb)

def FourMomentaFromHelicityAngles(md, ma, mb, theta, phi) :
  """
    Calculate the four-momenta of the decay products in D->AB in the rest frame of D
      md:    mass of D
      ma:    mass of A
      mb:    mass of B
      theta: angle between A momentum in D rest frame and D momentum in its helicity frame
      phi:   angle of plane formed by A & B in D helicity frame
  """
  # Calculate magnitude of momentum in D rest frame
  p = TwoBodyMomentum(md, ma, mb)
  # Calculate energy in D rest frame
  Ea = Sqrt(p**2 + ma**2)
  Eb = Sqrt(p**2 + mb**2)
  # Construct four-momenta with A aligned with D in D helicity frame
  Pa = LorentzVector(Vector(Zeros(p), Zeros(p),  p), Ea)
  Pb = LorentzVector(Vector(Zeros(p), Zeros(p), -p), Eb)
  # Rotate four-momenta
  Pa = RotateFourVector(Pa, Zeros(phi), -theta, -phi)
  Pb = RotateFourVector(Pb, Zeros(phi), -theta, -phi)
  return Pa,Pb

def RecursiveSum( vectors ):
  """
    Helper function fro CalculateHelicityAngles. It sums all the vectors in
    a list or nested list
  """
  return sum([RecursiveSum( vector ) if isinstance(vector,list) else vector for vector in vectors])

def CalculateHelicityAngles( pdecays ):
  """
  Calculate the Helicity Angles for every decay topology specified with brackets []
  examples:
     - input:
       A -> B (-> C D) E (-> F G) ==> CalculateHelicityAngles([[C,D],[F,G]])
       A -> B (-> C (-> D E) F) G ==> CalculateHelicityAngles([ [ [ D, E] , F ] , G ])
     - output:
       A -> B (-> C D) E (-> F G) ==> (thetaB,phiB,thetaC,phiC,thetaF,phiF)
       A -> B (-> C (-> D E) F) G ==> (thetaB,phiB,thetaC,phiC,thetaD,phiD)
       where thetaX,phiX are the polar and azimuthal angles of X in the mother rest frame
  """
  angles = ()
  if len(pdecays)!=2:
    sys.exit('ERROR in CalculateHelicityAngles: lenght of the input list is different from 2')
    
  for i,pdau in enumerate(pdecays):
    if i==0:
      angles += HelicityAngles( RecursiveSum(pdau) if isinstance(pdau,list) else pdau )
    if isinstance(pdau,list):# the particle is not basic but decay, rotate and boost to its new rest frame
      pmother = RecursiveSum(pdau)
      pdau_newframe = RotationAndBoost( pdau, pmother )
      angles += CalculateHelicityAngles( pdau_newframe )
  return angles

def ChangeAxes(ps, (xnew,ynew,znew) ):
  """
    Return a list of LorentzVector with the component described by the
    new axes (x,y,z).
  """
  pout = []
  for p in ps:
    px = XComponent(p)
    py = YComponent(p)
    pz = ZComponent(p)
    pout.append( LorentzVector( Vector( px*XComponent(xnew)+py*YComponent(xnew)+pz*ZComponent(xnew),
                                        px*XComponent(ynew)+py*YComponent(ynew)+pz*ZComponent(ynew),
                                        px*XComponent(znew)+py*YComponent(znew)+pz*ZComponent(znew) ) ,TimeComponent(p)) )
  return pout

def RotatedAxes(pb, oldaxes = None ) :
  """
    Return new (rotated) axes aligned with the momentum vector pb
  """
  z1 = UnitVector(SpatialComponents(pb))       # New z-axis is in the direction of pb
  eb = TimeComponent(pb)
  zeros = Zeros(eb)
  ones = Ones(eb)
  z0 = Vector(zeros, zeros, ones) if oldaxes==None else oldaxes[2]  # Old z-axis vector
  x0 = Vector(ones, zeros, zeros) if oldaxes==None else oldaxes[0]  # Old x-axis vector
  sp = ScalarProduct(z1, z0)
  a0 = z0 - z1*Scalar(sp)   # Vector in z-pb plane perpendicular to z0
  x1 = tf.where(tf.equal(sp, 1.0), x0, -UnitVector(a0))
  y1 = VectorProduct(z1, x1)                   # New y-axis
  return (x1, y1, z1)

def OldAxes(pb) :
  """
    Return old (before rotation) axes in the frame aligned with the momentum vector pb
  """
  z1 = UnitVector(SpatialComponents(pb))       # New z-axis is in the direction of pb
  eb = TimeComponent(pb)
  z0 = Vector(Zeros(eb), Zeros(eb), Ones(eb))  # Old z-axis vector
  x0 = Vector(Ones(eb), Zeros(eb), Zeros(eb))  # Old x-axis vector
  sp = ScalarProduct(z1, z0)
  a0 = z0 - z1*Scalar(sp)   # Vector in z-pb plane perpendicular to z0
  x1 = tf.where(tf.equal(sp, 1.0), x0, -UnitVector(a0))
  y1 = VectorProduct(z1, x1)                   # New y-axis
  x = Vector(XComponent(x1), XComponent(y1), XComponent(z1))
  y = Vector(YComponent(x1), YComponent(y1), YComponent(z1))
  z = Vector(ZComponent(x1), ZComponent(y1), ZComponent(z1))
  return (x, y, z)

def RotationAndBoost(ps, pb) :
  """
    Rotate and boost all momenta from the list ps to the rest frame of pb
    After the rotation, the coordinate system is defined as:
      z axis: direction of pb
      y axis: perpendicular to the plane formed by the old z and pb
      x axis: [y,z]

    ps : list of Lorentz vectors to rotate and boost
    pb : Lorentz vector defining the new frame
    Returns:
      ps1        : list of transformed Lorentz vectors
  """
  newaxes = RotatedAxes(pb)
  eb = TimeComponent(pb)
  zeros = Zeros(eb)
  boost = Vector(zeros, zeros, -Norm(SpatialComponents(pb))/eb) # Boost vector in the rotated coordinates along z axis

  return ApplyRotationAndBoost(ps,newaxes,boost)

def ApplyRotationAndBoost(ps,(x,y,z),boost):
  """
  Helper function for RotationAndBoost. It applies RotationAndBoost iteratively on nested lists
  """
  ps1 = []
  for p in ps : 
    p1 = ProjectFourVector(p, (x1, y1, z1))
    p2 = LorentzBoost(p1, boost)
    ps1 += [ p2 ]

  return ps1

def EulerAngles(x1, y1, z1, x2, y2, z2) : 
  """
    Calculate Euler angles (phi, theta, psi in the ZYZ convention) which transform the coordinate basis (x1, y1, z1)
    to the basis (x2, y2, z2). Both x1,y1,z1 and x2,y2,z2 are assumed to be orthonormal and right-handed.
  """
  theta = Acos(  ScalarProduct(z1, z2) )
  phi   = Atan2( ScalarProduct(z1, y2), ScalarProduct(z1, x2) )
  psi   = Atan2( ScalarProduct(y1, z2), ScalarProduct(x1, z2) )
  return (phi, theta, psi)

def HelicityAngles3Body(pa, pb, pc) :
  """
  Calculate 4 helicity angles for the 3-body D->ABC decay defined as:
    theta_r, phi_r : polar and azimuthal angles of the AB resonance in the D rest frame
    theta_a, phi_a : polar and azimuthal angles of the A in AB rest frame
  """
  theta_r = Acos( -ZComponent(pc) / Norm( SpatialComponents(pc) ) )
  phi_r = Atan2( -YComponent(pc), -XComponent(pc) )

  pa_prime = LorentzVector( RotateVector(SpatialComponents(pa), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pa) )
  pb_prime = LorentzVector( RotateVector(SpatialComponents(pb), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pb) )

  w = TimeComponent(pa) + TimeComponent(pb)

  pab = LorentzVector( -(pa_prime + pb_prime)/Scalar(w), w)
  pa_prime2 = LorentzBoost(pa_prime, pab)

  theta_a = Acos( -ZComponent(pa_prime2) / Norm( SpatialComponents(pa_prime2) ) )
  phi_a = Atan2( -YComponent(pa_prime2), -XComponent(pa_prime2) )

  return (theta_r, phi_r, theta_a, phi_a)

def CosHelicityAngle( p1, p2):
    '''
    The helicity angle is defined as the angle between one of the two momenta in the p1+p2 rest frame
    with respect to the momentum of the p1+p2 system in the decaying particle rest frame (ptot)
    '''
    p12 = LorentzVector( SpatialComponents(p1)+SpatialComponents(p2),
                         TimeComponent(p1)+TimeComponent(p2) )
    pcm1 = BoostToRest(p1, p12)
    cosHel = ScalarProduct(UnitVector(SpatialComponents(pcm1)),UnitVector(SpatialComponents(p12)))
    return cosHel

def Azimuthal4Body( p1, p2, p3, p4):
    '''
    Calculates the angle between the plane defined by (p1,p2) and (p3,p4)
    '''
    v1 = SpatialComponents(p1)
    v2 = SpatialComponents(p2)
    v3 = SpatialComponents(p3)
    v4 = SpatialComponents(p4)
    n12 = UnitVector(VectorProduct( v1, v2 ))
    n34 = UnitVector(VectorProduct( v3, v4 ))
    z = UnitVector(v1+v2)
    cosPhi = ScalarProduct(n12, n34)
    sinPhi = ScalarProduct( VectorProduct(n12,n34),z )
    phi = Atan2(sinPhi,cosPhi) # defined in [-pi,pi]
    return phi

def HelicityAngles4Body(pa, pb, pc, pd) :
  """
  Calculate 4 helicity angles for the 4-body E->ABCD decay defined as:
    theta_ab, phi_ab : polar and azimuthal angles of the AB resonance in the E rest frame
    theta_cd, phi_cd : polar and azimuthal angles of the CD resonance in the E rest frame
    theta_ac, phi_ac : polar and azimuthal angles of the AC resonance in the E rest frame
    theta_bd, phi_bd : polar and azimuthal angles of the BD resonance in the E rest frame
    theta_ad, phi_ad : polar and azimuthal angles of the AD resonance in the E rest frame
    theta_bc, phi_bc : polar and azimuthal angles of the BC resonance in the E rest frame
    phi_ab_cd : azimuthal angle between AB and CD
    phi_ac_bd : azimuthal angle between AC and BD
    phi_ad_bc : azimuthal angle between AD and BC
  """
  theta_r = Acos( -ZComponent(pc) / Norm( SpatialComponents(pc) ) )
  phi_r = Atan2( -YComponent(pc), -XComponent(pc) )

  pa_prime = LorentzVector( RotateVector(SpatialComponents(pa), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pa) )
  pb_prime = LorentzVector( RotateVector(SpatialComponents(pb), -phi_r, Pi()-theta_r, phi_r), TimeComponent(pb) )

  w = TimeComponent(pa) + TimeComponent(pb)

  pab = LorentzVector( -(pa_prime + pb_prime)/Scalar(w), w)
  pa_prime2 = LorentzBoost(pa_prime, pab)

  theta_a = Acos( -ZComponent(pa_prime2) / Norm( SpatialComponents(pa_prime2) ) )
  phi_a = Atan2( -YComponent(pa_prime2), -XComponent(pa_prime2) )

  return (theta_r, phi_r, theta_a, phi_a)

def WignerD(phi,theta,psi,j2,m2_1,m2_2):
  """
  Calculate Wigner capital-D function. 
    phi,
    theta,
    psi  : Rotation angles
    j : spin (in units of 1/2, e.g. 1 for spin=1/2)
    m1 and m2 : spin projections (in units of 1/2, e.g. 1 for projection 1/2)
  """
  i = Complex(Const(0),Const(1))
  m1 = m2_1/2.
  m2 = m2_2/2.
  return Exp(-i*CastComplex(m1*phi))*CastComplex(Wignerd(theta,j2,m2_1,m2_2))*Exp(-i*CastComplex(m2*psi))

def Wignerd(theta, j, m1, m2) :
  """
  Calculate Wigner small-d function. Needs sympy.
    theta : angle
    j : spin (in units of 1/2, e.g. 1 for spin=1/2)
    m1 and m2 : spin projections (in units of 1/2)
  """
  from sympy import Rational
  from sympy.abc import x
  from sympy.utilities.lambdify import lambdify
  from sympy.physics.quantum.spin import Rotation as Wigner
  d = Wigner.d(Rational(j,2), Rational(m1,2), Rational(m2,2), x).doit().evalf()
  return lambdify(x, d, "tensorflow")(theta)

def WignerdExplicit(theta, j, m1, m2) :
  """
  Calculate Wigner small-d function. Does not need sympy
    theta : angle
    j : spin (in units of 1/2, e.g. 1 for spin=1/2)
    m1 and m2 : spin projections (in units of 1/2)
  """
  # Half-integer spins
  if j ==  1  and m1 ==  -1  and m2 ==  -1  : return     Cos(theta/2)
  if j ==  1  and m1 ==  -1  and m2 ==   1  : return     Sin(theta/2)
  if j ==  1  and m1 ==   1  and m2 ==  -1  : return    -Sin(theta/2)
  if j ==  1  and m1 ==   1  and m2 ==   1  : return     Cos(theta/2)
  if j ==  3  and m1 ==  -3  and m2 ==  -3  : return     (1+Cos(theta))/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==  -3  and m2 ==  -1  : return     math.sqrt(3.)*(1+Cos(theta))/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==  -3  and m2 ==   1  : return     math.sqrt(3.)*(1-Cos(theta))/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==  -3  and m2 ==   3  : return     (1-Cos(theta))/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==  -1  and m2 ==  -3  : return    -math.sqrt(3.)*(1+Cos(theta))/2*Sin(theta/2)#ok
  #if j ==  3  and m1 ==  -1  and m2 ==  -1  : return     Cos(theta/2)/4  +  3*Cos(3*theta/2)/4
  #if j ==  3  and m1 ==  -1  and m2 ==   1  : return    -Sin(theta/2)/4  +  3*Sin(3*theta/2)/4
  if j ==  3  and m1 ==  -1  and m2 ==  -1  : return     (3*Cos(theta)-1)/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==  -1  and m2 ==   1  : return     (3*Cos(theta)+1)/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==  -1  and m2 ==   3  : return     math.sqrt(3.)*(1-Cos(theta))/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==   1  and m2 ==  -3  : return     math.sqrt(3.)*(1-Cos(theta))/2*Cos(theta/2)#ok
  #if j ==  3  and m1 ==   1  and m2 ==  -1  : return     Sin(theta/2)/4  -  3*Sin(3*theta/2)/4
  #if j ==  3  and m1 ==   1  and m2 ==   1  : return     Cos(theta/2)/4  +  3*Cos(3*theta/2)/4
  if j ==  3  and m1 ==   1  and m2 ==  -1  : return    -(3*Cos(theta)+1)/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==   1  and m2 ==   1  : return     (3*Cos(theta)-1)/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==   1  and m2 ==   3  : return     math.sqrt(3.)*(1+Cos(theta))/2*Sin(theta/2)#ok  
  if j ==  3  and m1 ==   3  and m2 ==  -3  : return    -(1-Cos(theta))/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==   3  and m2 ==  -1  : return     math.sqrt(3.)*(1-Cos(theta))/2*Cos(theta/2)#ok
  if j ==  3  and m1 ==   3  and m2 ==   1  : return    -math.sqrt(3.)*(1+Cos(theta))/2*Sin(theta/2)#ok
  if j ==  3  and m1 ==   3  and m2 ==   3  : return     (1+Cos(theta))/2*Cos(theta/2)#ok
  if j ==  5  and m1 ==  -1  and m2 ==  -1  : return     Cos(theta/2)/4  +    Cos(3*theta/2)/8  + 5*Cos(5*theta/2)/8
  if j ==  5  and m1 ==  -1  and m2 ==   1  : return     Sin(theta/2)/4  -    Sin(3*theta/2)/8  + 5*Sin(5*theta/2)/8
  if j ==  5  and m1 ==   1  and m2 ==  -1  : return    -Sin(theta/2)/4  +    Sin(3*theta/2)/8  - 5*Sin(5*theta/2)/8
  if j ==  5  and m1 ==   1  and m2 ==   1  : return     Cos(theta/2)/4  +    Cos(3*theta/2)/8  + 5*Cos(5*theta/2)/8
  if j ==  7  and m1 ==  -1  and m2 ==  -1  : return   9*Cos(theta/2)/64 + 15*Cos(3*theta/2)/64 + 5*Cos(5*theta/2)/64 + 35*Cos(7*theta/2)/64
  if j ==  7  and m1 ==  -1  and m2 ==   1  : return  -9*Sin(theta/2)/64 + 15*Sin(3*theta/2)/64 - 5*Sin(5*theta/2)/64 + 35*Sin(7*theta/2)/64
  if j ==  7  and m1 ==   1  and m2 ==  -1  : return   9*Sin(theta/2)/64 - 15*Sin(3*theta/2)/64 + 5*Sin(5*theta/2)/64 - 35*Sin(7*theta/2)/64
  if j ==  7  and m1 ==   1  and m2 ==   1  : return   9*Cos(theta/2)/64 + 15*Cos(3*theta/2)/64 + 5*Cos(5*theta/2)/64 + 35*Cos(7*theta/2)/64

  # Integer spins
  if j ==  0  and m1 ==  -2  and m2 ==  -2  : return  0
  if j ==  0  and m1 ==  -2  and m2 ==   0  : return  0
  if j ==  0  and m1 ==  -2  and m2 ==   2  : return  0
  if j ==  0  and m1 ==   0  and m2 ==  -2  : return  0
  if j ==  0  and m1 ==   0  and m2 ==   0  : return  1.
  if j ==  0  and m1 ==   0  and m2 ==   2  : return  0
  if j ==  0  and m1 ==   2  and m2 ==  -2  : return  0
  if j ==  0  and m1 ==   2  and m2 ==   0  : return  0
  if j ==  0  and m1 ==   2  and m2 ==   2  : return  0
  if j ==  2  and m1 ==  -2  and m2 ==  -2  : return  Cos(theta)/2. + 1/2.
  if j ==  2  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(2.)*Sin(theta)/2.
  if j ==  2  and m1 ==  -2  and m2 ==   2  : return  -Cos(theta)/2. + 1/2.
  if j ==  2  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(2.)*Sin(theta)/2.
  if j ==  2  and m1 ==   0  and m2 ==   0  : return  Cos(theta)
  if j ==  2  and m1 ==   0  and m2 ==   2  : return  math.sqrt(2.)*Sin(theta)/2.
  if j ==  2  and m1 ==   2  and m2 ==  -2  : return  -Cos(theta)/2. + 1/2.
  if j ==  2  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(2.)*Sin(theta)/2.
  if j ==  2  and m1 ==   2  and m2 ==   2  : return  Cos(theta)/2. + 1/2.
  if j ==  4  and m1 ==  -2  and m2 ==  -2  : return  Cos(theta)/2. + Cos(2*theta)/2.
  if j ==  4  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(6.)*Sin(2*theta)/4.
  if j ==  4  and m1 ==  -2  and m2 ==   2  : return  Cos(theta)/2. - Cos(2*theta)/2.
  if j ==  4  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(6.)*Sin(2*theta)/4.
  if j ==  4  and m1 ==   0  and m2 ==   0  : return  3.*Cos(2*theta)/4. + 1/4.
  if j ==  4  and m1 ==   0  and m2 ==   2  : return  math.sqrt(6.)*Sin(2*theta)/4.
  if j ==  4  and m1 ==   2  and m2 ==  -2  : return  Cos(theta)/2. - Cos(2*theta)/2.
  if j ==  4  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(6.)*Sin(2*theta)/4.
  if j ==  4  and m1 ==   2  and m2 ==   2  : return  Cos(theta)/2. + Cos(2*theta)/2.
  if j ==  6  and m1 ==  -2  and m2 ==  -2  : return  Cos(theta)/32. + 5.*Cos(2*theta)/16. + 15.*Cos(3*theta)/32. + 3/16.
  if j ==  6  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(3.)*(Sin(theta) + 5.*Sin(3*theta))/16.
  if j ==  6  and m1 ==  -2  and m2 ==   2  : return  -Cos(theta)/32. + 5.*Cos(2*theta)/16. - 15.*Cos(3*theta)/32. + 3/16.
  if j ==  6  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(3.)*(Sin(theta) + 5.*Sin(3*theta))/16.
  if j ==  6  and m1 ==   0  and m2 ==   0  : return  3.*Cos(theta)/8. + 5.*Cos(3*theta)/8.
  if j ==  6  and m1 ==   0  and m2 ==   2  : return  math.sqrt(3.)*(Sin(theta) + 5.*Sin(3*theta))/16.
  if j ==  6  and m1 ==   2  and m2 ==  -2  : return  -Cos(theta)/32. + 5.*Cos(2*theta)/16. - 15.*Cos(3*theta)/32. + 3/16.
  if j ==  6  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(3.)*(Sin(theta) + 5.*Sin(3*theta))/16.
  if j ==  6  and m1 ==   2  and m2 ==   2  : return  Cos(theta)/32. + 5.*Cos(2*theta)/16. + 15.*Cos(3*theta)/32. + 3/16.

  print "Error in Wignerd: j,m1,m2 = ", j, m1, m2

  return None

def SpinRotationAngle(pa, pb, pc, bachelor = 2) :
  """
    Calculate the angle between two spin-quantisation axes for the 3-body D->ABC decay
    aligned along the particle B and particle A.
      pa, pb, pc : 4-momenta of the final-state particles
      bachelor : index of the "bachelor" particle (0=A, 1=B, or 2=C)
  """
  if bachelor == 2 : return Const(0.)
  pboost = LorentzVector( -SpatialComponents(pb)/Scalar(TimeComponent(pb)), TimeComponent(pb))
  if bachelor == 0 :
    pa1 = SpatialComponents(LorentzBoost(pa, pboost))
    pc1 = SpatialComponents(LorentzBoost(pc, pboost))
    return Acos( ScalarProduct(pa1, pc1)/Norm(pa1)/Norm(pc1) )
  if bachelor == 1 :
    pac = pa + pc
    pac1 = SpatialComponents(LorentzBoost(pac, pboost))
    pa1  = SpatialComponents(LorentzBoost(pa, pboost))
    return Acos( ScalarProduct(pac1, pa1)/Norm(pac1)/Norm(pa1) )
  return None

def HelicityAmplitude3Body(thetaR, phiR, thetaA, phiA, spinD, spinR, mu, lambdaR, lambdaA, lambdaB, lambdaC, cache = False) :
  """
  Calculate complex helicity amplitude for the 3-body decay D->ABC
    thetaR, phiR : polar and azimuthal angles of AB resonance in D rest frame
    thetaA, phiA : polar and azimuthal angles of A in AB rest frame
    spinD : D spin
    spinR : spin of the intermediate R resonance
    mu : D spin projection onto z axis
    lambdaR : R resonance helicity
    lambdaA : A helicity
    lambdaB : B helicity
    lambdaC : C helicity
  """
  lambda1 = lambdaR - lambdaC
  lambda2 = lambdaA - lambdaB
  ph = (mu-lambda1)/2.*phiR + (lambdaR-lambda2)/2.*phiA
  d_terms = Wignerd(thetaR, spinD, mu, lambda1)*Wignerd(thetaA, spinR, lambdaR, lambda2)
  h = Complex(d_terms*Cos(ph), d_terms*Sin(ph))

  if cache : Optimisation.cacheable_tensors += [ h ]

  return h

def HelicityCouplingsFromLS(ja, jb, jc, lb, lc, bls) :
  """
    Return the helicity coupling from a list of LS couplings.
      ja : spin of A (decaying) particle
      jb : spin of B (1st decay product)
      jc : spin of C (2nd decay product)
      lb : B helicity
      lc : C helicity
      bls : dictionary of LS couplings, where:
        keys are tuples corresponding to (L,S) pairs
        values are values of LS couplings
    Note that ALL j,l,s should be doubled, e.g. S=1 for spin-1/2, L=2 for P-wave etc.
  """
  a = 0.
  for ls, b in bls.iteritems() :
    l = ls[0]
    s = ls[1]
    coeff = math.sqrt((l+1)/(ja+1))*Clebsch(jb, lb, jc, -lc, s, lb-lc)*Clebsch(l, 0, s, lb-lc, ja, lb-lc)
    a += Const(coeff)*b
  return a

def ZemachTensor(m2ab, m2ac, m2bc, m2d, m2a, m2b, m2c, spin, cache = False) :
  """
    Zemach tensor for 3-body D->ABC decay
  """
  z = None
  if spin == 0 : z = Complex( Const(1.), Const(0.))
  if spin == 1 : z = Complex( m2ac-m2bc+(m2d-m2c)*(m2b-m2a)/m2ab, Const(0.))
  if spin == 2 : z = Complex( (m2bc-m2ac+(m2d-m2c)*(m2a-m2b)/m2ab)**2-1./3.*(m2ab-2.*(m2d+m2c)+(m2d-m2c)**2/m2ab)*(m2ab-2.*(m2a+m2b)+(m2a-m2b)**2/m2ab), Const(0.))
  if cache : Optimisation.cacheable_tensors += [ z ]

  return z

def TwoBodyMomentum(md, ma, mb) :
  """
    Momentum of two-body decay products D->AB in the D rest frame
  """
  return Sqrt((md**2-(ma+mb)**2)*(md**2-(ma-mb)**2)/(4*md**2))

def ComplexTwoBodyMomentum(md, ma, mb) :
  """
    Momentum of two-body decay products D->AB in the D rest frame.
    Output value is a complex number, analytic continuation for the
    region below threshold.
  """
  return Sqrt(Complex((md**2-(ma+mb)**2)*(md**2-(ma-mb)**2)/(4*md**2), Const(0.)))

def FindBasicParticles(particle):
  if particle.GetNDaughters()==0: return [particle]
  basic_particles = []
  for dau in particle.GetDaughters():
    basic_particles += FindBasicParticles(dau)
  return basic_particles

def AllowedHelicities(particle):
  return range(-particle.GetSpin2(),particle.GetSpin2(),2)+[particle.GetSpin2()]

def HelicityMatrixDecayChain(parent,helAmps):
  matrix_parent =  HelicityMatrixElement(parent,helAmps)
  daughters = parent.GetDaughters()
  if all(dau.GetNDaughters()==0 for dau in daughters): return matrix_parent

  heldaug = list(itertools.product(*[AllowedHelicities(dau) for dau in daughters]))
  
  d1basics = FindBasicParticles(daughters[0])
  d2basics = FindBasicParticles(daughters[1])
  d1helbasics = list(itertools.product(*[AllowedHelicities(bas) for bas in d1basics]))
  d2helbasics = list(itertools.product(*[AllowedHelicities(bas) for bas in d2basics]))

  #matrix_dau = [HelicityMatrixDecayChain(dau,helAmps) for dau in daughters if dau.GetNDaughters()!=0 else {(d2hel,)+d2helbasic: c for d2hel,d2helbasic in itertools.product(AllowedHelicites(dau),d2helbasics)}]
  #matrix_dau=[]
  #for dau in daughters:
  #  if dau.GetNDaughters()!=0:
  #    matrix_dau.append( HelicityMatrixDecayChain(dau,helAmps) )
  matrix_dau = [HelicityMatrixDecayChain(dau,helAmps) for dau in daughters if dau.GetNDaughters()!=0]

  matrix={}
  for phel,d1helbasic,d2helbasic in itertools.product(AllowedHelicities(parent),d1helbasics,d2helbasics):
    if len(matrix_dau)==2:
      matrix[(phel,)+d1helbasic+d2helbasic]= sum([matrix_parent[(phel,d1hel,d2hel)]*matrix_dau[0][(d1hel,)+d1helbasic]*matrix_dau[1][(d2hel,)+d2helbasic] for d1hel,d2hel in heldaug if abs(parent.GetSpin2())>=abs(d1hel-d2hel)])
    elif daughters[0].GetNDaughters()!=0:
      matrix[(phel,)+d1helbasic+d2helbasic]= sum([matrix_parent[(phel,d1hel,d2hel)]*matrix_dau[0][(d1hel,)+d1helbasic] for d1hel,d2hel in heldaug if abs(parent.GetSpin2())>=abs(d1hel-d2hel)])
    else:
      matrix[(phel,)+d1helbasic+d2helbasic]= sum([matrix_parent[(phel,d1hel,d2hel)]*matrix_dau[0][(d2hel,)+d2helbasic] for d1hel,d2hel in heldaug if abs(parent.GetSpin2())>=abs(d1hel-d2hel)])
  return matrix

def HelicityMatrixElement(parent,helAmps):
  if parent.GetNDaughters()!=2:
    sys.exit('ERROR in HelicityMatrixElement, the parent '+parent.GetName()+' has no 2 daughters')

  matrixelement = {}
  [d1,d2] = parent.GetDaughters()
  parent_helicities = AllowedHelicities(parent)
  d1_helicities = AllowedHelicities(d1)
  d2_helicities = AllowedHelicities(d2)

  if parent.IsParityConserving():
    if not all(part.GetParity() in [-1,+1] for part in [parent,d1,d2]):
      sys.exit('ERROR in HelicityMatrixElement for the decay of particle '+parent.GetName()+\
               ', the parities have to be correctly defined (-1 or +1) for the particle and its daughters')

    parity_factor = parent.GetParity()*d1.GetParity()*d2.GetParity()*(-1)**((d1.GetSpin2()+d2.GetSpin2()-parent.GetSpin2())/2)
    if 0 in d1_helicities and 0 in d2_helicities and parity_factor==-1 \
       and helAmps[parent.GetName()+'_'+d1.GetName()+'_0_'+d2.GetName()+'_0']!=0:
      sys.exit('ERROR in HelicityMatrixElement, the helicity amplitude '\
               +parent.GetName()+'_'+d1.GetName()+'_0_'+d2.GetName()+'_0 should be set to 0 for parity conservation reason')

  theta = d1.Theta()
  phi = d1.Phi()
  for phel,d1hel,d2hel in itertools.product(parent_helicities,d1_helicities,d2_helicities):
    if parent.GetSpin2()<abs(d1hel-d2hel): continue
    d1hel_str = ('+' if d1hel>0 else '')+str(d1hel)
    d2hel_str = ('+' if d2hel>0 else '')+str(d2hel)
    flipped = False
    if parent.IsParityConserving() and ( d1hel!=0 or d2hel!=0):
      d1hel_str_flip = ('+' if -d1hel>0 else '')+str(-d1hel)
      d2hel_str_flip = ('+' if -d2hel>0 else '')+str(-d2hel)
      helAmp_str = parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str+'_'+d2.GetName()+'_'+d2hel_str
      helAmp_str_flip = parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str_flip+'_'+d2.GetName()+'_'+d2hel_str_flip
      if helAmp_str in helAmps.keys() and helAmp_str_flip in helAmps.keys():
        sys.exit('ERROR in HelicityMatrixElement: particle '+parent.GetName()+\
                 ' conserves parity in decay but both '+helAmp_str+' and '+helAmp_str_flip+\
                 ' are declared. Only one has to be declared, the other is calculated from parity conservation')
      if helAmp_str_flip in helAmps.keys():
        d1hel_str = d1hel_str_flip
        d2hel_str = d2hel_str_flip
        flipped = True
    helAmp = (parity_factor if parent.IsParityConserving() and flipped else 1.) * helAmps[parent.GetName()+'_'+d1.GetName()+'_'+d1hel_str+'_'+d2.GetName()+'_'+d2hel_str]
    matrixelement[(phel,d1hel,d2hel)] = parent.GetShape()*helAmp\
                                        *Conjugate(WignerD(phi,theta,0,parent.GetSpin2(),phel,d1hel-d2hel))
  return matrixelement

def RotateFinalStateHelicity(matrixin,particlesfrom,particlesto):
  if not all(part1.GetSpin2()==part2.GetSpin2() for part1,part2 in zip(particlesfrom,particlesto)):
    sys.exit('ERROR in RotateFinalStateHelicity, found a mismatch between the spins given to RotateFinalStateHelicity')
  matrixout = {}
  for hels in matrixin.keys(): matrixout[hels]=0
  heldaugs = []
  axesfrom = [RotatedAxes(part.GetMomentum(),oldaxes=part.GetAxes()) for part in particlesfrom]
  axesto = [RotatedAxes(part.GetMomentum(),oldaxes=part.GetAxes()) for part in particlesto]
  thetas = [Acos(ScalarProduct(axisfrom[2],axisto[2])) for axisfrom,axisto  in zip(axesfrom,axesto)]
  phis = [Atan2(ScalarProduct(axisfrom[1],axisto[0]),ScalarProduct(axisfrom[0],axisto[0])) for axisfrom,axisto  in zip(axesfrom,axesto)]
  
  rot = []
  for part,theta,phi in zip(particlesfrom,thetas,phis):
    allhels = AllowedHelicities(part)
    rot.append({})
    for helfrom,helto in itertools.product(allhels,allhels):
      rot[-1][(helfrom,helto)] = Conjugate(WignerD(phi,theta,0,part.GetSpin2(),helfrom,helto))
  
  for helsfrom in matrixin.keys():
    daughelsfrom = helsfrom[1:]
    for helsto in matrixout.keys():
      daughelsto = helsto[1:]
      prod = reduce(operator.mul,[rot[i][(hpfrom,hpto)] for i,(hpfrom,hpto) in enumerate(zip(daughelsfrom,daughelsto))])
      matrixout[helsto] += prod*matrixin[helsfrom]

  return matrixout

class Particle:
  """
  Class to describe a Particle
  """
  def __init__(self, name = 'default', shape = None, spin2 = 0, momentum=None, daughters = [], parityConserving=False, parity=None):
    self._name = name
    self._spin2 = spin2
    self._daughters = daughters
    self._shape = shape if shape!=None else CastComplex(Const(1.))
    if momentum!=None and daughters!=[]:
      sys.exit('ERROR in Particle '+name+' definition: do not define the momentum, it is taken from the sum of the daughters momenta!')
    self._momentum = momentum if momentum!=None and daughters==[] else sum([dau.GetMomentum() for dau in daughters])
    self._parityConserving = parityConserving
    self._parity = parity
    emom = TimeComponent(self._momentum)
    zeros = Zeros(emom)
    ones = Ones(emom)
    self._axes = ( Vector(ones,zeros,zeros),
                   Vector(zeros,ones,zeros),
                   Vector(zeros,zeros,ones) )

  def GetName(self): return self._name
  def GetSpin2(self): return self._spin2
  def GetDaughters(self): return self._daughters
  def GetNDaughters(self): return len(self._daughters)
  def GetShape(self): return self._shape
  def GetMomentum(self): return self._momentum
  def GetAxes(self): return self._axes
  def IsParityConserving(self): return self._parityConserving
  def GetParity(self): return self._parity
  def SetName(self,newname): self._name=newname
  def SetSpin(self,newspin): self._spin2=newspin
  def SetShape(self,newshape): self._shape=newshape
  def SetMomentum(self,momentum): self._momentum=momentum
  def SetParity(self,parity): self._parity=parity
  def Theta(self):
    return Acos( ScalarProduct(UnitVector(SpatialComponents(self._momentum)),self._axes[2]) )
  def Phi(self):
    x = self._axes[0]
    y = self._axes[1]
    return Atan2( ScalarProduct(UnitVector(SpatialComponents(self._momentum)),y), ScalarProduct(UnitVector(SpatialComponents(self._momentum)),x) )

  def ApplyRotationAndBoost(self,newaxes,boost):
    self._axes = newaxes
    self._momentum = LorentzBoost(self._momentum,boost)
    for dau in self._daughters: dau.ApplyRotationAndBoost(newaxes,boost)

  def RotateAndBoostDaughters(self, isAtRest=True ):
    if not isAtRest:
      newaxes = RotatedAxes(self._momentum, oldaxes = self._axes )
      eb = TimeComponent(self._momentum)
      zeros = Zeros(eb)
      boost = -SpatialComponents(self._momentum)/Scalar(eb)
      #boost = newaxes[2]*(-Norm(SpatialComponents(self._momentum))/eb)
      #boost = Vector(zeros, zeros, -Norm(SpatialComponents(self._momentum))/eb)
      for dau in self._daughters: dau.ApplyRotationAndBoost(newaxes,boost)
    for dau in self._daughters: dau.RotateAndBoostDaughters(isAtRest=False)

  def __eq__(self,other):
    eq = (self._name==other._name)
    eq &= (self._spin2==other._spin2)
    eq &= (self._shape==other._shape)
    eq &= (self._momentum==other._momentum)
    eq &= (self._daughters==other._daughters)
    return eq

class DalitzPhaseSpace :
  """
  Class for Dalitz plot (2D) phase space for the 3-body decay D->ABC
  """

  def __init__(self, ma, mb, mc, md, mabrange = None, mbcrange = None, macrange = None, symmetric = False ) :
    """
    Constructor
      ma - A mass
      mb - B mass
      mc - C mass
      md - D (mother) mass
    """
    self.ma = ma
    self.mb = mb
    self.mc = mc
    self.md = md
    self.ma2 = ma*ma
    self.mb2 = mb*mb
    self.mc2 = mc*mc
    self.md2 = md*md
    self.msqsum = self.md2 + self.ma2 + self.mb2 + self.mc2
    self.minab = (ma + mb)**2
    self.maxab = (md - mc)**2
    self.minbc = (mb + mc)**2
    self.maxbc = (md - ma)**2
    self.minac = (ma + mc)**2
    self.maxac = (md - mb)**2
    self.macrange = macrange
    self.symmetric = symmetric
    if mabrange :
      if mabrange[1]**2 < self.maxab : self.maxab = mabrange[1]**2
      if mabrange[0]**2 > self.minab : self.minab = mabrange[0]**2
    if mbcrange :
      if mbcrange[1]**2 < self.maxbc : self.maxbc = mbcrange[1]**2
      if mbcrange[0]**2 > self.minbc : self.maxbc = mbcrange[0]**2
    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")

  def Inside(self, x) :
    """
      Check if the point x=(M2ab, M2bc) is inside the phase space
    """
    m2ab = self.M2ab(x)
    m2bc = self.M2bc(x)
    mab = Sqrt(m2ab)

    inside = tf.logical_and(tf.logical_and(tf.greater(m2ab, self.minab), tf.less(m2ab, self.maxab)), \
                            tf.logical_and(tf.greater(m2bc, self.minbc), tf.less(m2bc, self.maxbc)))

    if self.macrange :
      m2ac = self.msqsum - m2ab - m2bc
      inside = tf.logical_and(inside, tf.logical_and(tf.greater(m2ac, self.macrange[0]**2), tf.less(m2ac, self.macrange[1]**2)))

    if self.symmetric :
      inside = tf.logical_and(inside, tf.greater( m2bc, m2ab ))

    eb = (m2ab - self.ma2 + self.mb2)/2./mab
    ec = (self.md2 - m2ab - self.mc2)/2./mab
    p2b = eb**2 - self.mb2
    p2c = ec**2 - self.mc2
    inside = tf.logical_and(inside, tf.logical_and(tf.greater(p2c, 0), tf.greater(p2b, 0)))
    pb = Sqrt(p2b)
    pc = Sqrt(p2c)
    e2bc = (eb+ec)**2
    m2bc_max = e2bc - (pb - pc)**2
    m2bc_min = e2bc - (pb + pc)**2
    return tf.logical_and(inside, tf.logical_and(tf.greater(m2bc, m2bc_min), tf.less(m2bc, m2bc_max) ) )

  def Filter(self, x) :
    return tf.boolean_mask(x, self.Inside(x) )

  def UnfilteredSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
    """
    v = [ np.random.uniform(self.minab, self.maxab, size ).astype('d'),
          np.random.uniform(self.minbc, self.maxbc, size ).astype('d') ]
    if majorant>0 : v += [ np.random.uniform( 0., majorant, size).astype('d') ]
    return np.transpose(np.array(v))

  def UniformSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    return self.Filter( self.UnfilteredSample(size, majorant) )

  def RectangularGridSample(self, sizeab, sizebc) :
    """
      Create a data sample in the form of rectangular grid of points within the phase space.
      Useful for normalisation.
        sizeab : number of grid nodes in M2ab range
        sizebc : number of grid nodes in M2bc range
    """
    size = sizeab*sizebc
    mgrid = np.lib.index_tricks.nd_grid()
    vab = mgrid[0:sizeab,0:sizebc][0]*(self.maxab-self.minab)/float(sizeab) + self.minab
    vbc = mgrid[0:sizeab,0:sizebc][1]*(self.maxbc-self.minbc)/float(sizebc) + self.minbc
    v = [ vab.reshape(size).astype('d'), vbc.reshape(size).astype('d') ]
    dlz = tf.stack(v , axis=1)
    return tf.boolean_mask(dlz, self.Inside(dlz) )

  def M2ab(self, sample) :
    """
      Return M2ab variable (vector) for the input sample
    """
    return sample[:,0]

  def M2bc(self, sample) :
    """
       Return M2bc variable (vector) for the input sample
    """
    return sample[:,1]

  def M2ac(self, sample) :
    """
      Return M2ac variable (vector) for the input sample.
      It is calculated from M2ab and M2bc
    """
    return self.msqsum - self.M2ab(sample) - self.M2bc(sample)

  def CosHelicityAB(self, sample) :
    """
      Calculate cos(helicity angle) of the AB resonance
    """
    return CosHelicityAngleDalitz(self.M2ab(sample), self.M2bc(sample), self.md, self.ma, self.mb, self.mc)

  def CosHelicityBC(self, sample) :
    """
       Calculate cos(helicity angle) of the BC resonance
    """
    return CosHelicityAngleDalitz(self.M2bc(sample), self.M2ac(sample), self.md, self.mb, self.mc, self.ma)

  def CosHelicityAC(self, sample) :
    """
       Calculate cos(helicity angle) of the AC resonance
    """
    return CosHelicityAngleDalitz(self.M2ac(sample), self.M2ab(sample), self.md, self.mc, self.ma, self.mb)

  def MPrimeAC(self, sample) :
    """
      Square Dalitz plot variable m'
    """
    mac = Sqrt(self.M2ac(sample))
    return Acos(2*(mac - math.sqrt(self.minac))/(math.sqrt(self.maxac) - math.sqrt(self.minac) ) - 1.)/math.pi

  def ThetaPrimeAC(self, sample) :
    """
      Square Dalitz plot variable theta'
    """
    return Acos(self.CosHelicityAC(sample))/math.pi

  def MPrimeAB(self, sample) :
    """
      Square Dalitz plot variable m'
    """
    mab = Sqrt(self.M2ab(sample))
    return Acos(2*(mab - math.sqrt(self.minab))/(math.sqrt(self.maxab) - math.sqrt(self.minab) ) - 1.)/math.pi

  def ThetaPrimeAB(self, sample) :
    """
      Square Dalitz plot variable theta'
    """
    return Acos(-self.CosHelicityAB(sample))/math.pi

  def MPrimeBC(self, sample) :
    """
      Square Dalitz plot variable m'
    """
    mbc = Sqrt(self.M2bc(sample))
    return Acos(2*(mbc - math.sqrt(self.minbc))/(math.sqrt(self.maxbc) - math.sqrt(self.minbc) ) - 1.)/math.pi

  def ThetaPrimeBC(self, sample) :
    """
      Square Dalitz plot variable theta'
    """
    return Acos(-self.CosHelicityBC(sample))/math.pi

  def Placeholder(self, name = None) :
    """
      Create a placeholder for a dataset in this phase space 
    """
    return tf.placeholder(fptype, shape = (None, None), name = name )

  def FromVectors(self, m2ab, m2bc) : 
    """
      Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
    """
    return tf.stack( [m2ab, m2bc], axis = 1 )

class DoubleDalitzPhaseSpace : 
  """
    Phase space representing two (correlated) Dalitz plots. 
  """
  def __init__(self, dlz1, dlz2) : 
    self.dlz1 = dlz1
    self.dlz2 = dlz2
    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")

  def Data1(self, x) : 
    return tf.slice(x, [0, 0], [-1, 2])

  def Data2(self, x) : 
    return tf.slice(x, [0, 2], [-1, 2])

  def Inside(self, x) : 
    return tf.logical_and(self.dlz1.Inside(self.Data1(x)), self.dlz2.Inside(self.Data2(x)))

  def Filter(self, x) : 
    return tf.boolean_mask(x, self.Inside(x) )

  def UnfilteredSample(self, size, majorant = -1) : 
    """
      Generate uniform sample of point within phase space. 
        size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                   so the number of points in the output will be <size. 
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                   uniform number from 0 to majorant. Useful for accept-reject toy MC. 
    """
    v = [ 
          np.random.uniform(self.dlz1.minab, self.dlz1.maxab, size ).astype('d'), 
          np.random.uniform(self.dlz1.minbc, self.dlz1.maxbc, size ).astype('d'), 
          np.random.uniform(self.dlz2.minab, self.dlz2.maxab, size ).astype('d'), 
          np.random.uniform(self.dlz2.minbc, self.dlz2.maxbc, size ).astype('d') 
        ]
    if majorant>0 : v += [ np.random.uniform( 0., majorant, size).astype('d') ]
    return np.transpose(np.array(v))

  def UniformSample(self, size, majorant = -1) : 
    """
      Generate uniform sample of point within phase space. 
        size     : number of _initial_ points to generate. Not all of them will fall into phase space, 
                   so the number of points in the output will be <size. 
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is 
                   uniform number from 0 to majorant. Useful for accept-reject toy MC. 
      Note it does not actually generate the sample, but returns the data flow graph for generation, 
      which has to be run within TF session. 
    """
    return self.Filter( self.UnfilteredSample(size, majorant) )

  def Placeholder(self, name = None) :
    """
      Create a placeholder for a dataset in this phase space 
    """
    return tf.placeholder(fptype, shape = (None, None), name = name )


class Baryonic3BodyPhaseSpace(DalitzPhaseSpace) :
  """
    Derived class for baryonic 3-body decay, baryon -> scalar scalar baryon
  """

  def FinalStateMomenta(self, m2ab, m2bc, thetab, phib, phiac) :
    """
      Calculate 4-momenta of final state tracks in the 5D phase space
        m2ab, m2bc : invariant masses of AB and BC combinations
        thetab, phib : direction angles of the particle B in the reference frame
        phiac : angle of AC plane wrt. polarisation plane
    """

    m2ac = self.msqsum - m2ab - m2bc

    p_a = TwoBodyMomentum(self.md, self.ma, Sqrt(m2bc))
    p_b = TwoBodyMomentum(self.md, self.mb, Sqrt(m2ac))
    p_c = TwoBodyMomentum(self.md, self.mc, Sqrt(m2ab))

    cos_theta_b = (p_a*p_a + p_b*p_b - p_c*p_c)/(2.*p_a*p_b)
    cos_theta_c = (p_a*p_a + p_c*p_c - p_b*p_b)/(2.*p_a*p_c)

    p4a = LorentzVector(Vector(Zeros(p_a), Zeros(p_a), p_a), Sqrt(p_a**2 + self.ma2) )
    p4b = LorentzVector(Vector( p_b*Sqrt(1. - cos_theta_b**2), Zeros(p_b), -p_b*cos_theta_b), Sqrt(p_b**2 + self.mb2) )
    p4c = LorentzVector(Vector(-p_c*Sqrt(1. - cos_theta_c**2), Zeros(p_c), -p_c*cos_theta_c), Sqrt(p_c**2 + self.mc2) )

    return (p4a, p4b, p4c)

class FourBodyAngularPhaseSpace :
  """
  Class for angular phase space of 4-body X->(AB)(CD) decay (3D).
  """

  def __init__(self, q2_min, q2_max, Bmass_window = [0.,6000.], Kpimass_window = [792.,992.]) :
    """
    Constructor
    """
    self.q2_min = q2_min
    self.q2_max = q2_max
    self.Bmass_min = Bmass_window[0]
    self.Bmass_max = Bmass_window[1]
    self.Kpimass_min = Kpimass_window[0]
    self.Kpimass_max = Kpimass_window[1]
    
    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")

  def Inside(self, x) :
    """
      Check if the point x=(cos_theta_1, cos_theta_2, phi) is inside the phase space
    """
    cos1 = self.CosTheta1(x)
    cos2 = self.CosTheta2(x)
    phi  = self.Phi(x)
    q2   = self.Q2(x)
    Bmass   = self.BMass(x)
    Kpimass = self.KpiMass(x)

    inside = tf.logical_and(tf.logical_and(tf.greater(cos1, -1.), tf.less(cos1, 1.)), \
                            tf.logical_and(tf.greater(cos2, -1.), tf.less(cos2, 1.)))
    inside = tf.logical_and(inside, \
                            tf.logical_and(tf.greater(phi, -math.pi), tf.less(phi, math.pi )))
    inside = tf.logical_and(inside, \
                            tf.logical_and(tf.greater(q2, self.q2_min), tf.less(q2, self.q2_max )))  
    inside = tf.logical_and(inside, tf.logical_and(tf.greater(Bmass, self.Bmass_min),
                                                   tf.less(Bmass, self.Bmass_max )))  
    inside = tf.logical_and(inside, tf.logical_and(tf.greater(Kpimass, self.Kpimass_min),
                                                   tf.less(Kpimass, self.Kpimass_max )))

    return inside

  def Filter(self, x) :
    return tf.boolean_mask(x, self.Inside(x) )

  def UnfilteredSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
    """
    v = [
          np.random.uniform(-1., 1., size ).astype('d'),
          np.random.uniform(-1., 1., size ).astype('d'),
          np.random.uniform(-math.pi, math.pi, size ).astype('d'),
          np.random.uniform(self.q2_min, self.q2_max, size ).astype('d'),
          np.random.uniform(self.Bmass_min, self.Bmass_max, size ).astype('d'),
          np.random.uniform(self.Kpimass_min, self.Kpimass_max, size ).astype('d')
        ]
    if majorant>0 : v += [ np.random.uniform( 0., majorant, size).astype('d') ]
    return np.transpose(np.array(v))

  def UniformSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    return self.Filter( self.UnfilteredSample(size, majorant) )

  def RectangularGridSample(self, size_cos_1, size_cos_2, size_phi, size_q2, size_Bmass, size_Kpimass) :
    """
      Create a data sample in the form of rectangular grid of points within the phase space.
      Useful for normalisation.
    """
    size = size_cos_1*size_cos_2*size_phi*size_q2*size_Bmass*size_Kpimass
    mgrid = np.lib.index_tricks.nd_grid()
    v1 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][0]*2./float(size_cos_1) - 1.
    v2 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][1]*2./float(size_cos_2) - 1.
    v3 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][2]*2.*math.pi/float(size_phi)
    v4 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][3]*(self.q2_max-self.q2_min)/float(size_q2) + self.q2_min
    v5 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][4]*\
        (self.Bmass_max - self.Bmass_min)/float(size_Bmass) + self.Bmass_min
    v6 = mgrid[0:size_cos_1,0:size_cos_2,0:size_phi,0:size_q2,0:size_Bmass,0:size_Kpimass][5]*\
        (self.Kpimass_max - self.Kpimass_min)/float(size_Kpimass) + self.Kpimass_min

    v = [ v1.reshape(size).astype('d'), v2.reshape(size).astype('d'), 
          v3.reshape(size).astype('d'), v4.reshape(size).astype('d'), 
          v5.reshape(size).astype('d'), v6.reshape(size).astype('d')]
    x = tf.stack(v , axis=1)
    return tf.boolean_mask(x, self.Inside(x) )

  def CosTheta1(self, sample) :
    """
      Return CosTheta1 variable (vector) for the input sample
    """
    return sample[:,0]

  def CosTheta2(self, sample) :
    """
      Return CosTheta2 variable (vector) for the input sample
    """
    return sample[:,1]

  def Phi(self, sample) :
    """
      Return Phi variable (vector) for the input sample
    """
    return sample[:,2]

  def Q2(self, sample) :
    """
      Return q^2 variable (vector) for the input sample
    """
    return sample[:,3]

  def BMass(self, sample) :
    """
      Return B-Mass variable (vector) for the input sample
    """
    return sample[:,4]

  def KpiMass(self, sample) :
    """
      Return Kpi-Mass variable (vector) for the input sample
    """
    return sample[:,5]
    

  def Placeholder(self, name = None) :
    return tf.placeholder(fptype, shape = (None, None), name = name )


class PHSPGenerator :
  def __init__(self, m_mother,  m_daughters) :
    """
      Constructor
    """
    self.ndaughters = len(m_daughters)
    self.m_daughters = m_daughters
    self.m_mother = m_mother

  def RandomOrdered(self, nev):
    return (-1)*tf.nn.top_k(-tf.random_uniform([nev,self.ndaughters-2],dtype=tf.float64),k=self.ndaughters-2).values

  def GenerateFlatAngles(self,nev):
    return ( tf.random_uniform([nev],minval=-1.,maxval=1.,dtype=tf.float64),
             tf.random_uniform([nev],minval=-math.pi,maxval=math.pi,dtype=tf.float64) )

  def RotateLorentzVector(self,p,costheta,phi):
    pvec = SpatialComponents(p)
    energy = TimeComponent(p)
    pvecrot = self.RotateVector(pvec,costheta,phi)
    return LorentzVector(pvecrot,energy)

  def RotateVector(self,vec, costheta,phi):
    cZ = costheta
    sZ = Sqrt(1-cZ**2)
    cY = Cos(phi)
    sY = Sin(phi)
    x = XComponent(vec)
    y = YComponent(vec)
    z = ZComponent(vec)
    xnew = cZ*x-sZ*y
    ynew = sZ*x+cZ*y
    xnew2 = cY*xnew-sY*z
    znew  = sY*xnew+cY*z
    return Vector(xnew2,ynew,znew)

  def GenerateModel(self, nev) :
    rands = self.RandomOrdered(nev)
    delta = self.m_mother-sum(self.m_daughters)

    sumsubmasses = []
    for i in range(self.ndaughters-2): sumsubmasses.append( sum(self.m_daughters[:(i+2)]) )
    SubMasses = rands*delta + sumsubmasses
    SubMasses = tf.concat([SubMasses,Scalar(self.m_mother*tf.ones([nev],dtype=tf.float64))],axis=1)
    pout = []
    weights = tf.ones([nev],dtype=tf.float64)
    for i in range(self.ndaughters-1):
      submass = tf.unstack(SubMasses,axis=1)[i]
      zeros = Zeros(submass)
      ones = Ones(submass)
      if i==0:
        MassDaughterA = self.m_daughters[i]*ones
        MassDaughterB = self.m_daughters[i+1]*ones
      else:
        MassDaughterA = tf.unstack(SubMasses,axis=1)[i-1]
        MassDaughterB = self.m_daughters[i+1]*Ones(MassDaughterA)
      pMag = TwoBodyMomentum(submass, MassDaughterA, MassDaughterB )
      (costheta,phi) = self.GenerateFlatAngles(nev)
      vecArot = self.RotateVector( Vector(zeros,pMag,zeros), costheta, phi )
      pArot = LorentzVector( vecArot, Sqrt( MassDaughterA**2+pMag**2 ) )
      pBrot = LorentzVector( -vecArot, Sqrt(MassDaughterB**2+pMag**2) )
      pout = [LorentzBoost(p, SpatialComponents(pArot)/Scalar(TimeComponent(pArot)) ) for p in pout]
      if i==0:
        pout.append(pArot)
        pout.append(pBrot)
      else: pout.append(pBrot)
      weights = tf.multiply(weights,pMag)
    moms = tf.concat(pout,axis=1)
    phsp_model = tf.concat([moms,Scalar(weights)],axis=1)
    return phsp_model

class NBody :
  """
  Class for N-body decay expressed as:
    m_mother   : mass of the mother
    m_daughs   : list of daughter masses
  """
  def __init__(self, m_mother, m_daughs ) :
    """
      Constructor
    """
    self.ndaughters=len(m_daughs)

    self.PHSPGenerator = PHSPGenerator(m_mother,m_daughs)
    self.nev_ph = tf.placeholder(tf.int32)
    self.majorant_ph = tf.placeholder(tf.float64)
    self.phsp_model = self.PHSPGenerator.GenerateModel(self.nev_ph)
    self.phsp_model_majorant = tf.concat([self.phsp_model,Scalar(tf.random_uniform([self.nev_ph],minval=0.,maxval=self.majorant_ph,dtype=fptype))],axis=1)

    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")

  def Filter(self, x):
    return x

  def Density(self, x) :
    return tf.transpose(x)[4*self.ndaughters]

  def UnfilteredSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    s = tf.Session()
    feed_dict = {self.nev_ph : size}
    if majorant>0: feed_dict.update({self.majorant_ph : majorant})
    uniform_sample = s.run( self.phsp_model_majorant if majorant > 0 else self.phsp_model,
                            feed_dict= feed_dict )
    s.close()
    return uniform_sample

  def FinalStateMomenta(self, x) :
    """
       Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
       defined by the phase space vector x. The momenta are calculated in the
       D rest frame.
    """
    p3s = [ Vector(tf.transpose(x)[4*i],tf.transpose(x)[4*i+1],tf.transpose(x)[4*i+2]) for i in range(self.ndaughters) ]
    pLs = [ LorentzVector(p3,tf.transpose(x)[4*i+3]) for p3,i in zip(p3s,range(self.ndaughters)) ]
    return tuple(pL for pL in pLs)

  def Placeholder(self, name = None) :
    return tf.placeholder(fptype, shape = (None, None), name = name )

'''
class FourBody :
  """
  Class for 4-body decay phase space D->(A1 A2)(B1 B2) expressed as:
    ma   : invariant mass of the A1 A2 combination
    mb   : invariant mass of the B1 B2 combination
    hela : cosine of the helicity angle of A1
    helb : cosine of the helicity angle of B1
    phi  : angle between the A1 A2 and B1 B2 planes in D rest frame
  """
  def __init__(self, ma1, ma2, mb1, mb2, md, useROOT=True ) :
    """
      Constructor
    """
    self.ma1 = ma1
    self.ma2 = ma2
    self.mb1 = mb1
    self.mb2 = mb2
    self.md = md
    self.ndaughters=4

    self.PHSPGenerator = PHSPGenerator(md,[ma1,ma2,mb1,mb2])
    self.nev_ph = tf.placeholder(tf.int32)
    self.majorant_ph = tf.placeholder(tf.float64)
    self.phsp_model = self.PHSPGenerator.GenerateModel(self.nev_ph)
    self.phsp_model_majorant = tf.concat([self.phsp_model,Scalar(tf.random_uniform([self.nev_ph],minval=0.,maxval=self.majorant_ph,dtype=fptype))],axis=1)

    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("data")

  def Density(self, x) :
    return tf.transpose(x)[4*self.ndaughters]

  def UniformSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    s = tf.Session()
    feed_dict = {self.nev_ph : size}
    if majorant>0: feed_dict.update({self.majorant_ph : majorant})
    uniform_sample = s.run( self.phsp_model_majorant if majorant > 0 else self.phsp_model,
                            feed_dict= feed_dict )
    s.close()
    return uniform_sample

  def FinalStateMomenta(self, x) :
    """
       Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
       defined by the phase space vector x. The momenta are calculated in the
       D rest frame.
    """
    p3s = [ Vector(tf.transpose(x)[4*i],tf.transpose(x)[4*i+1],tf.transpose(x)[4*i+2]) for i in range(self.ndaughters) ]
    pLs = [ LorentzVector(p3,tf.transpose(x)[4*i+3]) for p3,i in zip(p3s,range(self.ndaughters)) ]
    return tuple(pL for pL in pLs)

  def Placeholder(self, name = None) :
    return tf.placeholder(fptype, shape = (None, None), name = name )
'''

class FourBodyHelicityPhaseSpace :
  """
  Class for 4-body decay phase space D->(A1 A2)(B1 B2) expressed as:
    ma   : invariant mass of the A1 A2 combination
    mb   : invariant mass of the B1 B2 combination
    hela : cosine of the helicity angle of A1
    helb : cosine of the helicity angle of B1
    phi  : angle between the A1 A2 and B1 B2 planes in D rest frame
  """
  def __init__(self, ma1, ma2, mb1, mb2, md) :
    """
      Constructor
    """
    self.ma1 = ma1
    self.ma2 = ma2
    self.mb1 = mb1
    self.mb2 = mb2
    self.md = md

    self.ma1a2min = self.ma1 + self.ma2
    self.ma1a2max = self.md  - self.mb1 - self.mb2
    self.mb1b2min = self.mb1 + self.mb2
    self.mb1b2max = self.md  - self.ma1 - self.ma2

    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")

  def Inside(self, x) :
    """
      Check if the point x is inside the phase space
    """
    ma1a2 = self.Ma1a2(x)
    mb1b2 = self.Mb1b2(x)
    ctha  = self.CosHelicityA(x)
    cthb  = self.CosHelicityB(x)
    phi   = self.Phi(x)

    inside = tf.logical_and(tf.logical_and(tf.greater(ctha, -1.), tf.less(ctha, 1.)), \
                            tf.logical_and(tf.greater(cthb, -1.), tf.less(cthb, 1.)))
    inside = tf.logical_and(inside, \
                            tf.logical_and(tf.greater(phi, -math.pi ), tf.less(phi, math.pi ))
                           )

    mb1b2max = self.md - ma1a2

    inside = tf.logical_and(inside, tf.logical_and(tf.greater(ma1a2, self.ma1a2min), tf.less(ma1a2, self.ma1a2max)))
    inside = tf.logical_and(inside, tf.logical_and(tf.greater(mb1b2, self.mb1b2min), tf.less(mb1b2, mb1b2max)))

    return inside

  def Filter(self, x) :
    return tf.boolean_mask(x, self.Inside(x) )

  def Density(self, x) :
    ma1a2 = self.Ma1a2(x)
    mb1b2 = self.Mb1b2(x)
    d1 = TwoBodyMomentum(self.md, ma1a2, mb1b2)
    d2 = TwoBodyMomentum(ma1a2, self.ma1, self.ma2)
    d3 = TwoBodyMomentum(mb1b2, self.mb1, self.mb2)
    return d1*d2*d3/self.md

  def UnfilteredSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
    """
    v = [ np.random.uniform(self.ma1a2min, self.ma1a2max, size ).astype('d'),
          np.random.uniform(self.mb1b2min, self.mb1b2max, size ).astype('d'),
          np.random.uniform(-1., 1., size ).astype('d'),
          np.random.uniform(-1., 1., size ).astype('d'),
          np.random.uniform(-math.pi, math.pi, size ).astype('d'),
        ]
    if majorant>0 : v += [ np.random.uniform( 0., majorant, size).astype('d') ]
    return np.transpose(np.array(v))

  def UniformSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    return self.Filter( self.UnfilteredSample(size, majorant) )

  def Ma1a2(self, sample) :
    """
      Return M2ab variable (vector) for the input sample
    """
    return sample[:,0]

  def Mb1b2(self, sample) :
    """
      Return M2bc variable (vector) for the input sample
    """
    return sample[:,1]

  def CosHelicityA(self, sample) :
    """
      Return cos(helicity angle) of the A1A2 resonance
    """
    return sample[:,2]

  def CosHelicityB(self, sample) :
    """
       Return cos(helicity angle) of the B1B2 resonance
    """
    return sample[:,3]

  def Phi(self, sample) :
    """
       Return phi angle between A1A2 and B1B2 planes
    """
    return sample[:,4]

  def FinalStateMomenta(self, x) :
    """
       Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
       defined by the phase space vector x. The momenta are calculated in the
       D rest frame.
    """
    ma1a2  = self.Ma1a2(x)
    mb1b2  = self.Mb1b2(x)
    ctha   = self.CosHelicityA(x)
    cthb   = self.CosHelicityB(x)
    phi    = self.Phi(x)

    p0 = TwoBodyMomentum(self.md, ma1a2, mb1b2)
    pA = TwoBodyMomentum(ma1a2, self.ma1, self.ma2)
    pB = TwoBodyMomentum(mb1b2, self.mb1, self.mb2)

    zeros = Zeros(pA)

    p3A = RotateVector( Vector(zeros, zeros, pA), zeros, Acos(ctha), zeros)
    p3B = RotateVector( Vector(zeros, zeros, pB), zeros, Acos(cthb), phi)
    
    ea = Sqrt(p0**2 + ma1a2**2)
    eb = Sqrt(p0**2 + mb1b2**2)
    v0a = Vector(zeros, zeros,  p0/ea)
    v0b = Vector(zeros, zeros, -p0/eb)

    p4A1 = LorentzBoost(LorentzVector( p3A, Sqrt(self.ma1**2 + Norm(p3A)**2 ) ), v0a )
    p4A2 = LorentzBoost(LorentzVector(-p3A, Sqrt(self.ma2**2 + Norm(p3A)**2 ) ), v0a )
    p4B1 = LorentzBoost(LorentzVector( p3B, Sqrt(self.mb1**2 + Norm(p3B)**2 ) ), v0b )
    p4B2 = LorentzBoost(LorentzVector(-p3B, Sqrt(self.mb2**2 + Norm(p3B)**2 ) ), v0b )

    return (p4A1, p4A2, p4B1, p4B2)

  def Placeholder(self, name = None) :
    return tf.placeholder(fptype, shape = (None, None), name = name )

class RectangularPhaseSpace :
  """
  Class for rectangular phase space in n dimensions
  """

  def __init__(self, ranges = ((0., 1.)) ) :
    """
    Constructor
    """
    self.data_placeholder = self.Placeholder("data")
    self.norm_placeholder = self.Placeholder("norm")
    self.ranges = ranges

  def Inside(self, x) :
    """
      Check if the point x is inside the phase space
    """
    inside = tf.constant( [ True ], dtype = bool )
    for n,r in enumerate(self.ranges) :
      var = self.Coordinate(x, n)
      inside = tf.logical_and(inside, tf.logical_and(tf.greater(var, r[0]), tf.less(var, r[1])))
    return inside

  def Filter(self, x) :
    return tf.boolean_mask(x, self.Inside(x) )

  def UnfilteredSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
    """
    v = [ np.random.uniform(r[0], r[1], size ).astype('d') for r in self.ranges ]
    if majorant>0 : v += [ np.random.uniform( 0., majorant, size).astype('d') ]
    return np.transpose(np.array(v))

  def UniformSample(self, size, majorant = -1) :
    """
      Generate uniform sample of point within phase space.
        size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                   so the number of points in the output will be <size.
        majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                   uniform number from 0 to majorant. Useful for accept-reject toy MC.
      Note it does not actually generate the sample, but returns the data flow graph for generation,
      which has to be run within TF session.
    """
    return self.Filter( self.UnfilteredSample(size, majorant) )

  def RectangularGridSample(self, sizes) :
    """
      Create a data sample in the form of rectangular grid of points within the phase space.
      Useful for normalisation.
    """
    size = 1
    for i in sizes : size *= i
    v = []
    mg = np.mgrid[[slice(0,i) for i in sizes]]
    for i,(r,s) in enumerate(zip(self.ranges, sizes)) :
      v1 = (mg[i]+0.5)*(r[1]-r[0])/float(s) + r[0]
      v += [ v1.reshape(size).astype('d') ]
    x = tf.stack(v , axis=1)
    return tf.boolean_mask(x, self.Inside(x) )

  def Coordinate(self, sample, n) :
    """
      Return coordinate number n from the input sample
    """
    return sample[:,n]

  def Placeholder(self, name = None) :
    return tf.placeholder(fptype, shape = (None, None), name = name )

from __future__ import print_function, division, absolute_import


import sys
import operator
import math
import itertools

import tensorflow as tf
import numpy as np

from zfit.core import tfext
from zfit.core.utils import clebsch_coeff
from zfit.settings import fptype
from . import optimization


from functools import reduce  # py23: does that work?


def spatial_component(vector):
    """
    Return spatial components of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    #  return tf.slice(vector, [0, 0], [-1, 3])
    return vector[:, 0:3]


def time_component(vector):
    """
    Return time component of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    #  return tf.unstack(vector, axis=1)[3]
    return vector[:, 3]


def x_component(vector):
    """
    Return spatial X component of the input Lorentz or 3-vector
        vector : input vector
    """
    #  return tf.unstack(vector, axis=1)[0]
    return vector[:, 0]


def y_component(vector):
    """
    Return spatial Y component of the input Lorentz or 3-vector
        vector : input vector
    """
    return vector[:, 1]


def z_component(vector):
    """
    Return spatial Z component of the input Lorentz or 3-vector
        vector : input vector
    """
    return vector[:, 2]


def vector(x, y, z):
    """
    Make a 3-vector from components
        x, y, z : vector components
    """
    return tf.stack([x, y, z], axis=1)


def scalar(x):
    """
    Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
    One cannot do e.g. Const(2.)*vector(x, y, z), needs to do scalar(Const(2))*vector(x, y, z)
    """
    return tf.stack([x], axis=1)


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components
        space : 3-vector of spatial components
        time  : time component
    """
    return tf.concat([space, tf.stack([time], axis=1)], axis=1)


def metric_tensor():
    """
    Metric tensor for Lorentz space (constant)
    """
    return tf.constant([-1., -1., -1., 1.], dtype=fptype)


def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum
        vector : input Lorentz momentum vector
    """
    return tf.sqrt(tf.reduce_sum(vector * vector * metric_tensor(), 1))


def scalar_product(vec1, vec2):
    """
    Calculate scalar product of two 3-vectors
    """
    return tf.reduce_sum(vec1 * vec2, 1)


def vector_product(vec1, vec2):
    """
    Calculate vector product of two 3-vectors
    """
    return tf.cross(vec1, vec2)


def norm(vec):
    """
    Calculate norm of 3-vector
    """
    return tf.sqrt(tf.reduce_sum(vec * vec, 1))


def unit_vector(vec):
    """
    Return unit vector in the direction of vec
    """
    return vec / scalar(norm(vec))


def perp_unit_vector(vec1, vec2):
    """
    Return unit vector perpendicular to the plane formed by vec1 and vec2
    """
    v = vector_product(vec1, vec2)
    return v / scalar(norm(v))


def lorentz_boost(vector, boostvector):
    """
    Perform Lorentz boost
        vector :     4-vector to be boosted
        boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components
        are used)
    """
    boost = spatial_component(boostvector)
    b2 = scalar_product(boost, boost)
    gamma = 1. / tf.sqrt(1. - b2)
    gamma2 = (gamma - 1.0) / b2
    ve = time_component(vector)
    vp = spatial_component(vector)
    bp = scalar_product(vp, boost)
    vp2 = vp + scalar(gamma2 * bp + gamma * ve) * boost
    ve2 = gamma * (ve + bp)
    return lorentz_vector(vp2, ve2)


def boost_to_rest(vector, boostvector):
    """
    Perform Lorentz boost to the rest frame of the 4-vector boostvector.
    """
    boost = -spatial_component(boostvector) / scalar(time_component(boostvector))
    return lorentz_boost(vector, boost)


def boost_from_rest(vector, boostvector):
    """
    Perform Lorentz boost from the rest frame of the 4-vector boostvector.
    """
    boost = spatial_component(boostvector) / time_component(boostvector)
    return lorentz_boost(vector, boost)


def rotate_vector(v, phi, theta, psi):
    """
    Perform 3D rotation of the 3-vector
        v : vector to be rotated
        phi, theta, psi : Euler angles in Z-Y-Z convention
    """

    # Rotate Z (phi)
    c1 = tf.cos(phi)
    s1 = tf.sin(phi)
    c2 = tf.cos(theta)
    s2 = tf.sin(theta)
    c3 = tf.cos(psi)
    s3 = tf.sin(psi)

    # Rotate Y (theta)
    fzx2 = -s2 * c1
    fzy2 = s2 * s1
    fzz2 = c2

    # Rotate Z (psi)
    fxx3 = c3 * c2 * c1 - s3 * s1
    fxy3 = -c3 * c2 * s1 - s3 * c1
    fxz3 = c3 * s2
    fyx3 = s3 * c2 * c1 + c3 * s1
    fyy3 = -s3 * c2 * s1 + c3 * c1
    fyz3 = s3 * s2

    # Transform v
    vx = x_component(v)
    vy = y_component(v)
    vz = z_component(v)

    _vx = fxx3 * vx + fxy3 * vy + fxz3 * vz
    _vy = fyx3 * vx + fyy3 * vy + fyz3 * vz
    _vz = fzx2 * vx + fzy2 * vy + fzz2 * vz

    return vector(_vx, _vy, _vz)


def rotate_lorentz_vector(v, phi, theta, psi):
    """
    Perform 3D rotation of the 4-vector
        v : vector to be rotated
        phi, theta, psi : Euler angles in Z-Y-Z convention
    """
    return lorentz_vector(rotate_vector(spatial_component(v), phi, theta, psi), time_component(v))


def project_lorentz_vector(p, spatial):
    x1, y1, z1 = spatial
    p0 = spatial_component(p)
    p1 = lorentz_vector(vector(scalar_product(x1, p0), scalar_product(y1, p0), scalar_product(z1, p0)),
                        time_component(p))
    return p1


def cos_helicity_angle_dalitz(m2ab, m2bc, md, ma, mb, mc):
    """
    Calculate cos(helicity angle) for set of two Dalitz plot variables
        m2ab, m2bc : Dalitz plot variables (inv. masses squared of AB and BC combinations)
        md : mass of the decaying particle
        ma, mb, mc : masses of final state particles
    """
    md2 = md ** 2
    ma2 = ma ** 2
    mb2 = mb ** 2
    mc2 = mc ** 2
    m2ac = md2 + ma2 + mb2 + mc2 - m2ab - m2bc
    mab = tf.sqrt(m2ab)
    mac = tf.sqrt(m2ac)
    mbc = tf.sqrt(m2bc)
    p2a = 0.25 / md2 * (md2 - (mbc + ma) ** 2) * (md2 - (mbc - ma) ** 2)  # TODO: smell, unused
    p2b = 0.25 / md2 * (md2 - (mac + mb) ** 2) * (md2 - (mac - mb) ** 2)  # TODO: smell, unused
    p2c = 0.25 / md2 * (md2 - (mab + mc) ** 2) * (md2 - (mab - mc) ** 2)  # TODO: smell, unused
    eb = (m2ab - ma2 + mb2) / 2. / mab
    ec = (md2 - m2ab - mc2) / 2. / mab
    pb = tf.sqrt(eb ** 2 - mb2)
    pc = tf.sqrt(ec ** 2 - mc2)
    e2sum = (eb + ec) ** 2
    m2bc_max = e2sum - (pb - pc) ** 2
    m2bc_min = e2sum - (pb + pc) ** 2
    return (m2bc_max + m2bc_min - 2. * m2bc) / (m2bc_max - m2bc_min)


def spherical_angles(pb):
    """
        theta, phi : polar and azimuthal angles of the vector pb
    """
    z1 = unit_vector(spatial_component(pb))  # New z-axis is in the direction of pb
    theta = tf.acos(z_component(z1))  # Helicity angle
    phi = tf.atan2(y_component(pb), x_component(pb))  # phi angle
    return theta, phi


def helicity_angles(pb):
    """
    theta, phi : polar and azimuthal angles of the vector pb
    """
    return spherical_angles(pb)


def four_momentum_from_helicity_angles(md, ma, mb, theta, phi):
    """
    Calculate the four-momenta of the decay products in D->AB in the rest frame of D
      md:    mass of D
      ma:    mass of A
      mb:    mass of B
      theta: angle between A momentum in D rest frame and D momentum in its helicity frame
      phi:   angle of plane formed by A & B in D helicity frame
    """
    # Calculate magnitude of momentum in D rest frame
    p = two_body_momentum(md, ma, mb)
    # Calculate energy in D rest frame
    Ea = tf.sqrt(p ** 2 + ma ** 2)
    Eb = tf.sqrt(p ** 2 + mb ** 2)
    # Construct four-momenta with A aligned with D in D helicity frame
    Pa = lorentz_vector(vector(tf.zeros_like(p), tf.zeros_like(p), p), Ea)
    Pb = lorentz_vector(vector(tf.zeros_like(p), tf.zeros_like(p), -p), Eb)
    # Rotate four-momenta
    Pa = rotate_lorentz_vector(Pa, tf.zeros_like(phi), -theta, -phi)
    Pb = rotate_lorentz_vector(Pb, tf.zeros_like(phi), -theta, -phi)
    return Pa, Pb


def recursive_sum(vectors):
    """
    Helper function fro calc_helicity_angles. It sums all the vectors in
    a list or nested list
    """
    return sum([recursive_sum(vector) if isinstance(vector, list) else vector for vector in vectors])


def calc_helicity_angles(pdecays):
    """
    Calculate the Helicity Angles for every decay topology specified with brackets []
        examples:
        - input:
           A -> B (-> C D) E (-> F G) ==> calc_helicity_angles([[C,D],[F,G]])
           A -> B (-> C (-> D E) F) G ==> calc_helicity_angles([ [ [ D, E] , F ] , G ])
        - output:
           A -> B (-> C D) E (-> F G) ==> (thetaB,phiB,thetaC,phiC,thetaF,phiF)
           A -> B (-> C (-> D E) F) G ==> (thetaB,phiB,thetaC,phiC,thetaD,phiD)
        where thetaX,phiX are the polar and azimuthal angles of X in the mother rest frame
    """
    angles = ()
    if len(pdecays) != 2:
        sys.exit('ERROR in calc_helicity_angles: lenght of the input list is different from 2')

    for i, pdau in enumerate(pdecays):
        if i == 0:
            angles += helicity_angles(recursive_sum(pdau) if isinstance(pdau, list) else pdau)
        if isinstance(pdau,
                      list):  # the particle is not basic but decay, rotate and boost to its new
            # rest frame
            pmother = recursive_sum(pdau)
            pdau_newframe = rotate_and_boost(pdau, pmother)
            angles += calc_helicity_angles(pdau_newframe)
    return angles


def change_axis(ps, spatial_new):
    """
    Return a list of lorentz_vector with the component described by the
        new axes (x,y,z).
    """
    xnew, ynew, znew = spatial_new
    pout = []
    for p in ps:
        px = x_component(p)
        py = y_component(p)
        pz = z_component(p)
        pout.append(lorentz_vector(
            vector(px * x_component(xnew) + py * y_component(xnew) + pz * z_component(xnew),
                   px * x_component(ynew) + py * y_component(ynew) + pz * z_component(ynew),
                   px * x_component(znew) + py * y_component(znew) + pz * z_component(znew)),
            time_component(p)))
    return pout


def rotate_axis(pb, oldaxes=None):
    """
    Return new (rotated) axes aligned with the momentum vector pb
    """
    z1 = unit_vector(spatial_component(pb))  # New z-axis is in the direction of pb
    eb = time_component(pb)
    zeros = tf.zeros_like(eb)
    ones = tf.ones_like(eb)
    z0 = vector(zeros, zeros, ones) if oldaxes == None else oldaxes[2]  # Old z-axis vector
    x0 = vector(ones, zeros, zeros) if oldaxes == None else oldaxes[0]  # Old x-axis vector
    sp = scalar_product(z1, z0)
    a0 = z0 - z1 * scalar(sp)  # vector in z-pb plane perpendicular to z0
    x1 = tf.where(tf.equal(sp, 1.0), x0, -unit_vector(a0))
    y1 = vector_product(z1, x1)  # New y-axis
    return x1, y1, z1


def old_axis(pb):
    """
    Return old (before rotation) axes in the frame aligned with the momentum vector pb
    """
    z1 = unit_vector(spatial_component(pb))  # New z-axis is in the direction of pb
    eb = time_component(pb)
    z0 = vector(tf.zeros_like(eb), tf.zeros_like(eb), tf.ones_like(eb))  # Old z-axis vector
    x0 = vector(tf.ones_like(eb), tf.zeros_like(eb), tf.zeros_like(eb))  # Old x-axis vector
    sp = scalar_product(z1, z0)
    a0 = z0 - z1 * scalar(sp)  # vector in z-pb plane perpendicular to z0
    x1 = tf.where(tf.equal(sp, 1.0), x0, -unit_vector(a0))
    y1 = vector_product(z1, x1)  # New y-axis
    x = vector(x_component(x1), x_component(y1), x_component(z1))
    y = vector(y_component(x1), y_component(y1), y_component(z1))
    z = vector(z_component(x1), z_component(y1), z_component(z1))
    return x, y, z


def rotate_and_boost(ps, pb):
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
    newaxes = rotate_axis(pb)
    eb = time_component(pb)
    zeros = tf.zeros_like(eb)
    boost = vector(zeros, zeros, -norm(
        spatial_component(pb)) / eb)  # Boost vector in the rotated coordinates along z axis

    return apply_rotate_and_boost(ps, newaxes, boost)


def apply_rotate_and_boost(ps, spatial, boost):
    """
    Helper function for rotate_and_boost. It applies rotate_and_boost iteratively on nested lists
    """
    x, y, z = spatial
    ps1 = []
    for p in ps:
        p1 = project_lorentz_vector_(p, (x1, y1, z1))  # TODO: smell, should be x,y,z? are they lists?
        p2 = lorentz_boost(p1, boost)
        ps1 += [p2]

    return ps1


def euler_angles(x1, y1, z1, x2, y2, z2):
    """
    Calculate Euler angles (phi, theta, psi in the ZYZ convention) which transform the coordinate
    basis (x1, y1, z1)
    to the basis (x2, y2, z2). Both x1,y1,z1 and x2,y2,z2 are assumed to be orthonormal and
    right-handed.
    """
    theta = tf.acos(scalar_product(z1, z2))
    phi = tf.atan2(scalar_product(z1, y2), scalar_product(z1, x2))
    psi = tf.atan2(scalar_product(y1, z2), scalar_product(x1, z2))
    return phi, theta, psi


def helicity_angles_3body(pa, pb, pc):
    """
    Calculate 4 helicity angles for the 3-body D->ABC decay defined as:
    theta_r, phi_r : polar and azimuthal angles of the AB resonance in the D rest frame
    theta_a, phi_a : polar and azimuthal angles of the A in AB rest frame
    """
    theta_r = tf.acos(-z_component(pc) / norm(spatial_component(pc)))
    phi_r = tf.atan2(-y_component(pc), -x_component(pc))

    pa_prime = lorentz_vector(rotate_vector(spatial_component(pa), -phi_r, tfext.pi - theta_r, phi_r),
                              time_component(pa))
    pb_prime = lorentz_vector(rotate_vector(spatial_component(pb), -phi_r, tfext.pi - theta_r, phi_r),
                              time_component(pb))

    w = time_component(pa) + time_component(pb)

    pab = lorentz_vector(-(pa_prime + pb_prime) / scalar(w), w)
    pa_prime2 = lorentz_boost(pa_prime, pab)

    theta_a = tf.acos(-z_component(pa_prime2) / norm(spatial_component(pa_prime2)))
    phi_a = tf.atan2(-y_component(pa_prime2), -x_component(pa_prime2))

    return theta_r, phi_r, theta_a, phi_a


def cos_helicity_angle(p1, p2):
    """
    The helicity angle is defined as the angle between one of the two momenta in the p1+p2 rest
    frame
    with respect to the momentum of the p1+p2 system in the decaying particle rest frame (ptot)
    """
    p12 = lorentz_vector(spatial_component(p1) + spatial_component(p2),
                         time_component(p1) + time_component(p2))
    pcm1 = boost_to_rest(p1, p12)
    cosHel = scalar_product(unit_vector(spatial_component(pcm1)), unit_vector(spatial_component(p12)))
    return cosHel


def azimuthal_4body(p1, p2, p3, p4):
    """
    Calculates the angle between the plane defined by (p1,p2) and (p3,p4)
    """
    v1 = spatial_component(p1)
    v2 = spatial_component(p2)
    v3 = spatial_component(p3)
    v4 = spatial_component(p4)
    n12 = unit_vector(vector_product(v1, v2))
    n34 = unit_vector(vector_product(v3, v4))
    z = unit_vector(v1 + v2)
    cosPhi = scalar_product(n12, n34)
    sinPhi = scalar_product(vector_product(n12, n34), z)
    phi = tf.atan2(sinPhi, cosPhi)  # defined in [-pi,pi]
    return phi


def helicity_angles_4body(pa, pb, pc, pd):
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
    theta_r = tf.acos(-z_component(pc) / norm(spatial_component(pc)))
    phi_r = tf.atan2(-y_component(pc), -x_component(pc))

    pa_prime = lorentz_vector(rotate_vector(spatial_component(pa), -phi_r, tfext.pi - theta_r, phi_r),
                              time_component(pa))
    pb_prime = lorentz_vector(rotate_vector(spatial_component(pb), -phi_r, tfext.pi - theta_r, phi_r),
                              time_component(pb))

    w = time_component(pa) + time_component(pb)

    pab = lorentz_vector(-(pa_prime + pb_prime) / scalar(w), w)
    pa_prime2 = lorentz_boost(pa_prime, pab)

    theta_a = tf.acos(-z_component(pa_prime2) / norm(spatial_component(pa_prime2)))
    phi_a = tf.atan2(-y_component(pa_prime2), -x_component(pa_prime2))

    return theta_r, phi_r, theta_a, phi_a


def wigned_D(phi, theta, psi, j2, m2_1, m2_2):
    """
    Calculate Wigner capital-D function.
    phi,
    theta,
    psi  : Rotation angles
    j : spin (in units of 1/2, e.g. 1 for spin=1/2)
    m1 and m2 : spin projections (in units of 1/2, e.g. 1 for projection 1/2)
    """
    i = tf.complex(tfext.constant(0), tfext.constant(1))
    m1 = m2_1 / 2.
    m2 = m2_2 / 2.
    return tf.exp(-i * tfext.to_complex(m1 * phi)) * tfext.to_complex(
        wigner_d(theta, j2, m2_1, m2_2)) * tf.exp(-i * tfext.to_complex(m2 * psi))


def wigner_d(theta, j, m1, m2):
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
    d = Wigner.d(Rational(j, 2), Rational(m1, 2), Rational(m2, 2), x).doit().evalf()
    return lambdify(x, d, "tensorflow")(theta)


def wigner_explicit(theta, j, m1, m2):
    """
    Calculate Wigner small-d function. Does not need sympy
        theta : angle
        j : spin (in units of 1/2, e.g. 1 for spin=1/2)
        m1 and m2 : spin projections (in units of 1/2)
    """
    # Half-integer spins

    if j ==  1  and m1 ==  -1  and m2 ==  -1  : return     tf.cos(theta/2)
    if j ==  1  and m1 ==  -1  and m2 ==   1  : return     tf.sin(theta/2)
    if j ==  1  and m1 ==   1  and m2 ==  -1  : return    -tf.sin(theta/2)
    if j ==  1  and m1 ==   1  and m2 ==   1  : return     tf.cos(theta/2)
    if j ==  3  and m1 ==  -3  and m2 ==  -3  : return     (1+tf.cos(theta))/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==  -3  and m2 ==  -1  : return     math.sqrt(3.)*(1+tf.cos(theta))/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==  -3  and m2 ==   1  : return     math.sqrt(3.)*(1-tf.cos(theta))/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==  -3  and m2 ==   3  : return     (1-tf.cos(theta))/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==  -1  and m2 ==  -3  : return    -math.sqrt(3.)*(1+tf.cos(theta))/2*tf.sin(theta/2)#ok
    #if j ==  3  and m1 ==  -1  and m2 ==  -1  : return     tf.cos(theta/2)/4  +  3*tf.cos(3*theta/2)/4
    #if j ==  3  and m1 ==  -1  and m2 ==   1  : return    -tf.sin(theta/2)/4  +  3*tf.sin(3*theta/2)/4
    if j ==  3  and m1 ==  -1  and m2 ==  -1  : return     (3*tf.cos(theta)-1)/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==  -1  and m2 ==   1  : return     (3*tf.cos(theta)+1)/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==  -1  and m2 ==   3  : return     math.sqrt(3.)*(1-tf.cos(theta))/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==   1  and m2 ==  -3  : return     math.sqrt(3.)*(1-tf.cos(theta))/2*tf.cos(theta/2)#ok
    #if j ==  3  and m1 ==   1  and m2 ==  -1  : return     tf.sin(theta/2)/4  -  3*tf.sin(3*theta/2)/4
    #if j ==  3  and m1 ==   1  and m2 ==   1  : return     tf.cos(theta/2)/4  +  3*tf.cos(3*theta/2)/4
    if j ==  3  and m1 ==   1  and m2 ==  -1  : return    -(3*tf.cos(theta)+1)/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==   1  and m2 ==   1  : return     (3*tf.cos(theta)-1)/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==   1  and m2 ==   3  : return     math.sqrt(3.)*(1+tf.cos(theta))/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==   3  and m2 ==  -3  : return    -(1-tf.cos(theta))/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==   3  and m2 ==  -1  : return     math.sqrt(3.)*(1-tf.cos(theta))/2*tf.cos(theta/2)#ok
    if j ==  3  and m1 ==   3  and m2 ==   1  : return    -math.sqrt(3.)*(1+tf.cos(theta))/2*tf.sin(theta/2)#ok
    if j ==  3  and m1 ==   3  and m2 ==   3  : return     (1+tf.cos(theta))/2*tf.cos(theta/2)#ok
    if j ==  5  and m1 ==  -1  and m2 ==  -1  : return     tf.cos(theta/2)/4  +    tf.cos(3*theta/2)/8  + 5*tf.cos(5*theta/2)/8
    if j ==  5  and m1 ==  -1  and m2 ==   1  : return     tf.sin(theta/2)/4  -    tf.sin(3*theta/2)/8  + 5*tf.sin(5*theta/2)/8
    if j ==  5  and m1 ==   1  and m2 ==  -1  : return    -tf.sin(theta/2)/4  +    tf.sin(3*theta/2)/8  - 5*tf.sin(5*theta/2)/8
    if j ==  5  and m1 ==   1  and m2 ==   1  : return     tf.cos(theta/2)/4  +    tf.cos(3*theta/2)/8  + 5*tf.cos(5*theta/2)/8
    if j ==  7  and m1 ==  -1  and m2 ==  -1  : return   9*tf.cos(theta/2)/64 + 15*tf.cos(3*theta/2)/64 + 5*tf.cos(5*theta/2)/64 + 35*tf.cos(7*theta/2)/64
    if j ==  7  and m1 ==  -1  and m2 ==   1  : return  -9*tf.sin(theta/2)/64 + 15*tf.sin(3*theta/2)/64 - 5*tf.sin(5*theta/2)/64 + 35*tf.sin(7*theta/2)/64
    if j ==  7  and m1 ==   1  and m2 ==  -1  : return   9*tf.sin(theta/2)/64 - 15*tf.sin(3*theta/2)/64 + 5*tf.sin(5*theta/2)/64 - 35*tf.sin(7*theta/2)/64
    if j ==  7  and m1 ==   1  and m2 ==   1  : return   9*tf.cos(theta/2)/64 + 15*tf.cos(3*theta/2)/64 + 5*tf.cos(5*theta/2)/64 + 35*tf.cos(7*theta/2)/64

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
    if j ==  2  and m1 ==  -2  and m2 ==  -2  : return  tf.cos(theta)/2. + 1/2.
    if j ==  2  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(2.)*tf.sin(theta)/2.
    if j ==  2  and m1 ==  -2  and m2 ==   2  : return  -tf.cos(theta)/2. + 1/2.
    if j ==  2  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(2.)*tf.sin(theta)/2.
    if j ==  2  and m1 ==   0  and m2 ==   0  : return  tf.cos(theta)
    if j ==  2  and m1 ==   0  and m2 ==   2  : return  math.sqrt(2.)*tf.sin(theta)/2.
    if j ==  2  and m1 ==   2  and m2 ==  -2  : return  -tf.cos(theta)/2. + 1/2.
    if j ==  2  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(2.)*tf.sin(theta)/2.
    if j ==  2  and m1 ==   2  and m2 ==   2  : return  tf.cos(theta)/2. + 1/2.
    if j ==  4  and m1 ==  -2  and m2 ==  -2  : return  tf.cos(theta)/2. + tf.cos(2*theta)/2.
    if j ==  4  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(6.)*tf.sin(2*theta)/4.
    if j ==  4  and m1 ==  -2  and m2 ==   2  : return  tf.cos(theta)/2. - tf.cos(2*theta)/2.
    if j ==  4  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(6.)*tf.sin(2*theta)/4.
    if j ==  4  and m1 ==   0  and m2 ==   0  : return  3.*tf.cos(2*theta)/4. + 1/4.
    if j ==  4  and m1 ==   0  and m2 ==   2  : return  math.sqrt(6.)*tf.sin(2*theta)/4.
    if j ==  4  and m1 ==   2  and m2 ==  -2  : return  tf.cos(theta)/2. - tf.cos(2*theta)/2.
    if j ==  4  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(6.)*tf.sin(2*theta)/4.
    if j ==  4  and m1 ==   2  and m2 ==   2  : return  tf.cos(theta)/2. + tf.cos(2*theta)/2.
    if j ==  6  and m1 ==  -2  and m2 ==  -2  : return  tf.cos(theta)/32. + 5.*tf.cos(2*theta)/16. + 15.*tf.cos(3*theta)/32. + 3/16.
    if j ==  6  and m1 ==  -2  and m2 ==   0  : return  math.sqrt(3.)*(tf.sin(theta) + 5.*tf.sin(3*theta))/16.
    if j ==  6  and m1 ==  -2  and m2 ==   2  : return  -tf.cos(theta)/32. + 5.*tf.cos(2*theta)/16. - 15.*tf.cos(3*theta)/32. + 3/16.
    if j ==  6  and m1 ==   0  and m2 ==  -2  : return  -math.sqrt(3.)*(tf.sin(theta) + 5.*tf.sin(3*theta))/16.
    if j ==  6  and m1 ==   0  and m2 ==   0  : return  3.*tf.cos(theta)/8. + 5.*tf.cos(3*theta)/8.
    if j ==  6  and m1 ==   0  and m2 ==   2  : return  math.sqrt(3.)*(tf.sin(theta) + 5.*tf.sin(3*theta))/16.
    if j ==  6  and m1 ==   2  and m2 ==  -2  : return  -tf.cos(theta)/32. + 5.*tf.cos(2*theta)/16. - 15.*tf.cos(3*theta)/32. + 3/16.
    if j ==  6  and m1 ==   2  and m2 ==   0  : return  -math.sqrt(3.)*(tf.sin(theta) + 5.*tf.sin(3*theta))/16.
    if j ==  6  and m1 ==   2  and m2 ==   2  : return  tf.cos(theta)/32. + 5.*tf.cos(2*theta)/16. + 15.*tf.cos(3*theta)/32. + 3/16.


    print("Error in wigner_d: j,m1,m2 = ", j, m1, m2)


def spin_rotation_angle(pa, pb, pc, bachelor=2):
    """
    Calculate the angle between two spin-quantisation axes for the 3-body D->ABC decay
    aligned along the particle B and particle A.
      pa, pb, pc : 4-momenta of the final-state particles
      bachelor : index of the "bachelor" particle (0=A, 1=B, or 2=C)
    """
    if bachelor == 2:
        return tfext.constant(0.)
    pboost = lorentz_vector(-spatial_component(pb) / scalar(time_component(pb)), time_component(pb))
    if bachelor == 0:
        pa1 = spatial_component(lorentz_boost(pa, pboost))
        pc1 = spatial_component(lorentz_boost(pc, pboost))
        return tf.acos(scalar_product(pa1, pc1) / norm(pa1) / norm(pc1))
    if bachelor == 1:
        pac = pa + pc
        pac1 = spatial_component(lorentz_boost(pac, pboost))
        pa1 = spatial_component(lorentz_boost(pa, pboost))
        return tf.acos(scalar_product(pac1, pa1) / norm(pac1) / norm(pa1))
    return None


def helicity_amplitude_3body(thetaR, phiR, thetaA, phiA, spinD, spinR, mu, lambdaR, lambdaA, lambdaB,
                             lambdaC, cache=False):
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
    ph = (mu - lambda1) / 2. * phiR + (lambdaR - lambda2) / 2. * phiA
    d_terms = wigner_d(thetaR, spinD, mu, lambda1) * wigner_d(thetaA, spinR, lambdaR, lambda2)
    h = tf.complex(d_terms * tf.cos(ph), d_terms * tf.sin(ph))

    if cache:
        optimization.cacheable_tensors += [h]

    return h


def helicity_couplings_from_LS(ja, jb, jc, lb, lc, bls):
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
    for ls, b in bls.items():
        l = ls[0]
        s = ls[1]
        coeff = (math.sqrt((l + 1) / (ja + 1)) * clebsch_coeff(jb, lb, jc, -lc, s, lb - lc) *
                 clebsch_coeff(l, 0, s, lb - lc, ja, lb - lc))
        a += tfext.constant(coeff) * b
    return a


def zemach_tensor(m2ab, m2ac, m2bc, m2d, m2a, m2b, m2c, spin, cache=False):
    """
    Zemach tensor for 3-body D->ABC decay
    """
    z = None
    if spin == 0:
        z = tf.complex(tfext.constant(1.), tfext.constant(0.))
    if spin == 1:
        z = tf.complex(m2ac - m2bc + (m2d - m2c) * (m2b - m2a) / m2ab, tfext.constant(0.))
    if spin == 2:
        z = tf.complex((m2bc - m2ac + (m2d - m2c) * (m2a - m2b) / m2ab) ** 2 - 1. / 3. * (
            m2ab - 2. * (m2d + m2c) + (m2d - m2c) ** 2 / m2ab) * (
                           m2ab - 2. * (m2a + m2b) + (m2a - m2b) ** 2 / m2ab), tfext.constant(0.))
    if cache:
        optimization.cacheable_tensors += [z]

    return z


def two_body_momentum(md, ma, mb, calc_complex=False):
    """Momentum of two-body decay products D->AB in the D rest frame.

    Args:
        calc_complex (bool): If true, the output value is a complex number,
        allowing analytic continuation for the region below threshold.
    """
    squared_sum = (md ** 2 - (ma + mb) ** 2) * (md ** 2 - (ma - mb) ** 2) / (4 * md ** 2)
    if complex:
        squared_sum = tfext.to_complex(squared_sum)
    return tf.sqrt(squared_sum)


def find_basic_particles(particle):
    if particle.get_n_daughters() == 0:
        return [particle]
    basic_particles = []
    for dau in particle.get_daughters():
        basic_particles += find_basic_particles(dau)
    return basic_particles


def allowed_helicities(particle):
    return list(range(-particle.get_spin2(), particle.get_spin2(), 2)) + [particle.get_spin2()]


def helicity_matrix_decay_chain(parent, helAmps):
    matrix_parent = helicity_matrix_element(parent, helAmps)
    daughters = parent.get_daughters()
    if all(dau.get_n_daughters() == 0 for dau in daughters):
        return matrix_parent

    heldaug = itertools.product(*[allowed_helicities(dau) for dau in daughters])

    d1basics = find_basic_particles(daughters[0])
    d2basics = find_basic_particles(daughters[1])
    d1helbasics = itertools.product(*[allowed_helicities(bas) for bas in d1basics])
    d2helbasics = itertools.product(*[allowed_helicities(bas) for bas in d2basics])

    # matrix_dau = [helicity_matrix_decay_chain(dau,helAmps) for dau in daughters if
    # dau.get_n_daughters()!=0 else {(d2hel,)+d2helbasic: c for d2hel,d2helbasic in
    # itertools.product(AllowedHelicites(dau),d2helbasics)}]
    # matrix_dau=[]
    # for dau in daughters:
    #  if dau.get_n_daughters()!=0:
    #    matrix_dau.append( helicity_matrix_decay_chain(dau,helAmps) )
    matrix_dau = [helicity_matrix_decay_chain(dau, helAmps) for dau in daughters if dau.get_n_daughters() != 0]

    matrix = {}
    for phel, d1helbasic, d2helbasic in itertools.product(allowed_helicities(parent), d1helbasics,
                                                          d2helbasics):
        if len(matrix_dau) == 2:
            matrix[(phel,) + d1helbasic + d2helbasic] = sum([matrix_parent[(phel, d1hel, d2hel)] *
                                                             matrix_dau[0][(d1hel,) + d1helbasic] *
                                                             matrix_dau[1][(d2hel,) + d2helbasic]
                                                             for d1hel, d2hel in heldaug if
                                                             abs(parent.get_spin2()) >= abs(
                                                                 d1hel - d2hel)])
        elif daughters[0].get_n_daughters() != 0:
            matrix[(phel,) + d1helbasic + d2helbasic] = sum(
                [matrix_parent[(phel, d1hel, d2hel)] * matrix_dau[0][(d1hel,) + d1helbasic] for
                 d1hel, d2hel in heldaug if abs(parent.get_spin2()) >= abs(d1hel - d2hel)])
        else:
            matrix[(phel,) + d1helbasic + d2helbasic] = sum(
                [matrix_parent[(phel, d1hel, d2hel)] * matrix_dau[0][(d2hel,) + d2helbasic] for
                 d1hel, d2hel in heldaug if abs(parent.get_spin2()) >= abs(d1hel - d2hel)])
    return matrix


def helicity_matrix_element(parent, helAmps):
    if parent.get_n_daughters() != 2:
        sys.exit(
            'ERROR in helicity_matrix_element, the parent ' + parent.get_name() + ' has no 2 '
                                                                               'daughters')

    matrixelement = {}
    [d1, d2] = parent.get_daughters()
    parent_helicities = allowed_helicities(parent)
    d1_helicities = allowed_helicities(d1)
    d2_helicities = allowed_helicities(d2)

    if parent.is_parity_conserving():
        if not all(part.GetParity() in [-1, +1] for part in [parent, d1, d2]):
            sys.exit(
                'ERROR in helicity_matrix_element for the decay of particle ' + parent.get_name() +
                ', the parities have to be correctly defined (-1 or +1) for the particle and its '
                'daughters')

        parity_factor = parent.get_parity() * d1.get_parity() * d2.get_parity() * (-1) ** (
            (d1.get_spin2() + d2.get_spin2() - parent.get_spin2()) / 2)
        if 0 in d1_helicities and 0 in d2_helicities and parity_factor == -1 \
            and helAmps[parent.get_name() + '_' + d1.get_name() + '_0_' + d2.get_name() + '_0'] != 0:
            sys.exit('ERROR in helicity_matrix_element, the helicity amplitude ' \
                     + parent.get_name() + '_' + d1.get_name() + '_0_' + d2.get_name() +
                     '_0 should be set to 0 for parity conservation reason')

    theta = d1.theta()
    phi = d1.phi()
    for phel, d1hel, d2hel in itertools.product(parent_helicities, d1_helicities, d2_helicities):
        if parent.get_spin2() < abs(d1hel - d2hel):
            continue
        d1hel_str = ('+' if d1hel > 0 else '') + str(d1hel)
        d2hel_str = ('+' if d2hel > 0 else '') + str(d2hel)
        flipped = False
        if parent.is_parity_conserving() and (d1hel != 0 or d2hel != 0):
            d1hel_str_flip = ('+' if -d1hel > 0 else '') + str(-d1hel)
            d2hel_str_flip = ('+' if -d2hel > 0 else '') + str(-d2hel)
            helAmp_str = parent.get_name() + '_' + d1.get_name() + '_' + d1hel_str + '_' + \
                         d2.get_name() + '_' + d2hel_str
            helAmp_str_flip = parent.get_name() + '_' + d1.get_name() + '_' + d1hel_str_flip + '_' \
                              + d2.get_name() + '_' + d2hel_str_flip
            if helAmp_str in helAmps.keys() and helAmp_str_flip in helAmps.keys():
                sys.exit('ERROR in helicity_matrix_element: particle ' + parent.get_name() + \
                         ' conserves parity in decay but both ' + helAmp_str + ' and ' +
                         helAmp_str_flip + \
                         ' are declared. Only one has to be declared, the other is calculated '
                         'from parity conservation')
            if helAmp_str_flip in helAmps.keys():
                d1hel_str = d1hel_str_flip
                d2hel_str = d2hel_str_flip
                flipped = True
        helAmp = (parity_factor if parent.is_parity_conserving() and flipped else 1.) * helAmps[
            parent.get_name() + '_' + d1.get_name() + '_' + d1hel_str + '_' + d2.get_name() + '_' +
            d2hel_str]
        matrixelement[(phel, d1hel, d2hel)] = parent.get_shape() * helAmp \
                                              * tf.conj(wigned_D(phi, theta, 0,
                                                                 parent.get_spin2(),
                                                                 phel, d1hel - d2hel))
    return matrixelement


def rotate_final_state_helicity(matrixin, particlesfrom, particlesto):
    if not all(
        part1.get_spin2() == part2.get_spin2() for part1, part2 in zip(particlesfrom, particlesto)):
        sys.exit(
            'ERROR in rotate_final_state_helicity, found a mismatch between the spins given to '
            'rotate_final_state_helicity')
    matrixout = {}
    for hels in matrixin.keys():
        matrixout[hels] = 0
    heldaugs = []  # TODO: smell, unused
    axesfrom = [rotate_axis(part.get_momentum(), oldaxes=part.get_axis()) for part in particlesfrom]
    axesto = [rotate_axis(part.get_momentum(), oldaxes=part.get_axis()) for part in particlesto]
    thetas = [tf.acos(scalar_product(axisfrom[2], axisto[2])) for axisfrom, axisto in
              zip(axesfrom, axesto)]
    phis = [tf.atan2(scalar_product(axisfrom[1], axisto[0]), scalar_product(axisfrom[0], axisto[0]))
            for axisfrom, axisto in zip(axesfrom, axesto)]

    rot = []
    for part, theta, phi in zip(particlesfrom, thetas, phis):
        allhels = allowed_helicities(part)
        rot.append({})
        for helfrom, helto in itertools.product(allhels, allhels):
            rot[-1][(helfrom, helto)] = tf.conj(
                wigned_D(phi, theta, 0, part.get_spin2(), helfrom, helto))

    for helsfrom in matrixin.keys():
        daughelsfrom = helsfrom[1:]
        for helsto in matrixout.keys():
            daughelsto = helsto[1:]
            prod = reduce(operator.mul, [rot[i][(hpfrom, hpto)] for i, (hpfrom, hpto) in
                                         enumerate(zip(daughelsfrom, daughelsto))])
            matrixout[helsto] += prod * matrixin[helsfrom]

    return matrixout


class Particle(object):
    """
    Class to describe a Particle
    """

    def __init__(self, name='default', shape=None, spin2=0, momentum=None, daughters=[],
                 parity_conserving=False, parity=None):
        self._name = name
        self._spin2 = spin2
        self._daughters = daughters
        self._shape = shape if shape != None else tfext.to_complex(tfext.constant(1.))
        if momentum is not None and daughters != []:
            sys.exit(
                'ERROR in Particle ' + name + ' definition: do not define the momentum, '
                                              'it is taken from the sum of the daughters momenta!')
        self._momentum = momentum if momentum != None and daughters == [] else sum(
            [dau.get_momentum() for dau in daughters])
        self._parityConserving = parity_conserving
        self._parity = parity
        emom = time_component(self._momentum)
        zeros = tf.zeros_like(emom)
        ones = tf.ones_like(emom)
        self._axes = (vector(ones, zeros, zeros),
                      vector(zeros, ones, zeros),
                      vector(zeros, zeros, ones))

    def get_name(self):
        return self._name

    def get_spin2(self):
        return self._spin2

    def get_daughters(self):
        return self._daughters

    def get_n_daughters(self):
        return len(self._daughters)

    def get_shape(self):
        return self._shape

    def get_momentum(self):
        return self._momentum

    def get_axis(self):
        return self._axes

    def is_parity_conserving(self):
        return self._parityConserving

    def get_parity(self):
        return self._parity

    def set_name(self, newname):
        self._name = newname

    def set_spin(self, newspin):
        self._spin2 = newspin

    def set_shape(self, newshape):
        self._shape = newshape

    def set_momentum(self, momentum):
        self._momentum = momentum

    def set_parity(self, parity):
        self._parity = parity

    def theta(self):
        return tf.acos(scalar_product(unit_vector(spatial_component(self._momentum)), self._axes[2]))

    def phi(self):
        x = self._axes[0]
        y = self._axes[1]
        return tf.atan2(scalar_product(unit_vector(spatial_component(self._momentum)), y),
                        scalar_product(unit_vector(spatial_component(self._momentum)), x))

    def apply_rotate_and_boost(self, newaxes, boost):
        self._axes = newaxes
        self._momentum = lorentz_boost(self._momentum, boost)
        for dau in self._daughters:
            dau.apply_rotate_and_boost(newaxes, boost)

    def RotateAndBoostDaughters(self, isAtRest=True):
        if not isAtRest:
            newaxes = rotate_axis(self._momentum, oldaxes=self._axes)
            eb = time_component(self._momentum)
            zeros = tf.zeros_like(eb)
            boost = -spatial_component(self._momentum) / scalar(eb)
            # boost = newaxes[2]*(-Norm(spatial_component(self._momentum))/eb)
            # boost = vector(zeros, zeros, -Norm(spatial_component(self._momentum))/eb)
            for dau in self._daughters:
                dau.apply_rotate_and_boost(newaxes, boost)
        for dau in self._daughters:
            dau.RotateAndBoostDaughters(isAtRest=False)

    def __eq__(self, other):
        eq = (self._name == other._name)
        eq &= (self._spin2 == other._spin2)
        eq &= (self._shape == other._shape)
        eq &= (self._momentum == other._momentum)
        eq &= (self._daughters == other._daughters)
        return eq


class DalitzPhaseSpace(object):
    """
    Class for Dalitz plot (2D) phase space for the 3-body decay D->ABC
    """

    def __init__(self, ma, mb, mc, md, mabrange=None, mbcrange=None, macrange=None,
                 symmetric=False):
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
        self.ma2 = ma * ma
        self.mb2 = mb * mb
        self.mc2 = mc * mc
        self.md2 = md * md
        self.msqsum = self.md2 + self.ma2 + self.mb2 + self.mc2
        self.minab = (ma + mb) ** 2
        self.maxab = (md - mc) ** 2
        self.minbc = (mb + mc) ** 2
        self.maxbc = (md - ma) ** 2
        self.minac = (ma + mc) ** 2
        self.maxac = (md - mb) ** 2
        self.macrange = macrange
        self.symmetric = symmetric
        if mabrange:
            if mabrange[1] ** 2 < self.maxab:
                self.maxab = mabrange[1] ** 2
            if mabrange[0] ** 2 > self.minab:
                self.minab = mabrange[0] ** 2
        if mbcrange:
            if mbcrange[1] ** 2 < self.maxbc:
                self.maxbc = mbcrange[1] ** 2
            if mbcrange[0] ** 2 > self.minbc:
                self.maxbc = mbcrange[0] ** 2
        self.data_placeholder = self.placeholder("data")
        self.norm_placeholder = self.placeholder("norm")

    def inside(self, x):
        """
        Check if the point x=(M2ab, M2bc) is inside the phase space
        """
        m2ab = self.M2ab(x)
        m2bc = self.M2bc(x)
        mab = tf.sqrt(m2ab)

        inside = tf.logical_and(
            tf.logical_and(tf.greater(m2ab, self.minab), tf.less(m2ab, self.maxab)),
            tf.logical_and(tf.greater(m2bc, self.minbc), tf.less(m2bc, self.maxbc)))

        if self.macrange:
            m2ac = self.msqsum - m2ab - m2bc
            inside = tf.logical_and(inside, tf.logical_and(tf.greater(m2ac, self.macrange[0] ** 2),
                                                           tf.less(m2ac, self.macrange[1] ** 2)))

        if self.symmetric:
            inside = tf.logical_and(inside, tf.greater(m2bc, m2ab))

        eb = (m2ab - self.ma2 + self.mb2) / 2. / mab
        ec = (self.md2 - m2ab - self.mc2) / 2. / mab
        p2b = eb ** 2 - self.mb2
        p2c = ec ** 2 - self.mc2
        inside = tf.logical_and(inside, tf.logical_and(tf.greater(p2c, 0), tf.greater(p2b, 0)))
        pb = tf.sqrt(p2b)
        pc = tf.sqrt(p2c)
        e2bc = (eb + ec) ** 2
        m2bc_max = e2bc - (pb - pc) ** 2
        m2bc_min = e2bc - (pb + pc) ** 2
        return tf.logical_and(inside,
                              tf.logical_and(tf.greater(m2bc, m2bc_min), tf.less(m2bc, m2bc_max)))

    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [np.random.uniform(self.minab, self.maxab, size).astype('d'),
             np.random.uniform(self.minbc, self.maxbc, size).astype('d')]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def uniform_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, majorant))

    def rectangular_grid_sample(self, sizeab, sizebc):
        """
        Create a data sample in the form of rectangular grid of points within the phase space.
        Useful for normalisation.
            sizeab : number of grid nodes in M2ab range
            sizebc : number of grid nodes in M2bc range
        """
        size = sizeab * sizebc
        mgrid = np.lib.index_tricks.nd_grid()
        vab = mgrid[0:sizeab, 0:sizebc][0] * (self.maxab - self.minab) / float(sizeab) + self.minab
        vbc = mgrid[0:sizeab, 0:sizebc][1] * (self.maxbc - self.minbc) / float(sizebc) + self.minbc
        v = [vab.reshape(size).astype('d'), vbc.reshape(size).astype('d')]
        dlz = tf.stack(v, axis=1)
        return tf.boolean_mask(dlz, self.inside(dlz))

    def M2ab(self, sample):
        """
        Return M2ab variable (vector) for the input sample
        """
        return sample[:, 0]

    def M2bc(self, sample):
        """
        Return M2bc variable (vector) for the input sample
        """
        return sample[:, 1]

    def M2ac(self, sample):
        """
        Return M2ac variable (vector) for the input sample.
        It is calculated from M2ab and M2bc
        """
        return self.msqsum - self.M2ab(sample) - self.M2bc(sample)

    def cos_helicity_AB(self, sample):
        """
        Calculate cos(helicity angle) of the AB resonance
        """
        return cos_helicity_angle_dalitz(self.M2ab(sample), self.M2bc(sample), self.md, self.ma,
                                         self.mb, self.mc)

    def cos_helicity_BC(self, sample):
        """
        Calculate cos(helicity angle) of the BC resonance
        """
        return cos_helicity_angle_dalitz(self.M2bc(sample), self.M2ac(sample), self.md, self.mb,
                                         self.mc, self.ma)

    def CosHelicityAC(self, sample):
        """
        Calculate cos(helicity angle) of the AC resonance
        """
        return cos_helicity_angle_dalitz(self.M2ac(sample), self.M2ab(sample), self.md, self.mc,
                                         self.ma, self.mb)

    def M_prime_AC(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mac = tf.sqrt(self.M2ac(sample))
        return tf.acos(2 * (mac - math.sqrt(self.minac)) / (
            math.sqrt(self.maxac) - math.sqrt(self.minac)) - 1.) / math.pi

    def ThetaPrimeAC(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return tf.acos(self.CosHelicityAC(sample)) / math.pi

    def MPrimeAB(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mab = tf.sqrt(self.M2ab(sample))
        return tf.acos(2 * (mab - math.sqrt(self.minab)) / (
            math.sqrt(self.maxab) - math.sqrt(self.minab)) - 1.) / math.pi

    def ThetaPrimeAB(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return tf.acos(-self.cos_helicity_AB(sample)) / math.pi

    def MPrimeBC(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mbc = tf.sqrt(self.M2bc(sample))
        return tf.acos(2 * (mbc - math.sqrt(self.minbc)) / (
            math.sqrt(self.maxbc) - math.sqrt(self.minbc)) - 1.) / math.pi

    def ThetaPrimeBC(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return tf.acos(-self.cos_helicity_BC(sample)) / math.pi

    def placeholder(self, name=None):
        """
        Create a placeholder for a dataset in this phase space
        """
        return tf.placeholder(fptype, shape=(None, None), name=name)

    def from_vectors(self, m2ab, m2bc):
        """
        Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
        """
        return tf.stack([m2ab, m2bc], axis=1)


class DoubleDalitzPhaseSpace(object):
    """
    Phase space representing two (correlated) Dalitz plots.
    """

    def __init__(self, dlz1, dlz2):
        self.dlz1 = dlz1
        self.dlz2 = dlz2
        self.data_placeholder = self.placeholder("data")
        self.norm_placeholder = self.placeholder("norm")

    def data1(self, x):
        return tf.slice(x, [0, 0], [-1, 2])

    def data2(self, x):
        return tf.slice(x, [0, 2], [-1, 2])

    def inside(self, x):
        return tf.logical_and(self.dlz1.inside(self.data1(x)), self.dlz2.inside(self.data2(x)))

    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [
            np.random.uniform(self.dlz1.minab, self.dlz1.maxab, size).astype('d'),
            np.random.uniform(self.dlz1.minbc, self.dlz1.maxbc, size).astype('d'),
            np.random.uniform(self.dlz2.minab, self.dlz2.maxab, size).astype('d'),
            np.random.uniform(self.dlz2.minbc, self.dlz2.maxbc, size).astype('d')
            ]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def uniform_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, majorant))

    def placeholder(self, name=None):
        """
        Create a placeholder for a dataset in this phase space
        """
        return tf.placeholder(fptype, shape=(None, None), name=name)


class Baryonic3BodyPhaseSpace(DalitzPhaseSpace):
    """
    Derived class for baryonic 3-body decay, baryon -> scalar scalar baryon
    """

    def final_state_momenta(self, m2ab, m2bc, thetab, phib,
                            phiac):  # TODO: smell, unused thetab, phib, phiac
        """
        Calculate 4-momenta of final state tracks in the 5D phase space
            m2ab, m2bc : invariant masses of AB and BC combinations
            thetab, phib : direction angles of the particle B in the reference frame
            phiac : angle of AC plane wrt. polarisation plane
        """

        m2ac = self.msqsum - m2ab - m2bc

        p_a = two_body_momentum(self.md, self.ma, tf.sqrt(m2bc))
        p_b = two_body_momentum(self.md, self.mb, tf.sqrt(m2ac))
        p_c = two_body_momentum(self.md, self.mc, tf.sqrt(m2ab))

        cos_theta_b = (p_a * p_a + p_b * p_b - p_c * p_c) / (2. * p_a * p_b)
        cos_theta_c = (p_a * p_a + p_c * p_c - p_b * p_b) / (2. * p_a * p_c)

        p4a = lorentz_vector(vector(tf.zeros_like(p_a), tf.zeros_like(p_a), p_a),
                             tf.sqrt(p_a ** 2 + self.ma2))
        p4b = lorentz_vector(
            vector(p_b * tf.sqrt(1. - cos_theta_b ** 2), tf.zeros_like(p_b), -p_b * cos_theta_b),
            tf.sqrt(p_b ** 2 + self.mb2))
        p4c = lorentz_vector(
            vector(-p_c * tf.sqrt(1. - cos_theta_c ** 2), tf.zeros_like(p_c), -p_c * cos_theta_c),
            tf.sqrt(p_c ** 2 + self.mc2))

        return p4a, p4b, p4c


class FourBodyAngularPhaseSpace(object):
    """Class for angular phase space of 4-body X->(AB)(CD) decay (3D)."""

    def __init__(self, q2_min, q2_max, Bmass_window=[0., 6000.], Kpimass_window=[792., 992.]):
        self.q2_min = q2_min
        self.q2_max = q2_max
        self.Bmass_min = Bmass_window[0]
        self.Bmass_max = Bmass_window[1]
        self.Kpimass_min = Kpimass_window[0]
        self.Kpimass_max = Kpimass_window[1]

        self.data_placeholder = self.placeholder("data")
        self.norm_placeholder = self.placeholder("norm")

    def inside(self, x):
        """
        Check if the point x=(cos_theta_1, cos_theta_2, phi) is inside the phase space
        """
        cos1 = self.cos_theta_1(x)
        cos2 = self.cos_theta_2(x)
        phi = self.phi(x)
        q2 = self.Q2(x)
        Bmass = self.B_mass(x)
        Kpimass = self.Kpi_mass(x)

        inside = tf.logical_and(tf.logical_and(tf.greater(cos1, -1.), tf.less(cos1, 1.)),
                                tf.logical_and(tf.greater(cos2, -1.), tf.less(cos2, 1.)))
        inside = tf.logical_and(inside,
                                tf.logical_and(tf.greater(phi, -math.pi), tf.less(phi, math.pi)))
        inside = tf.logical_and(inside,
                                tf.logical_and(tf.greater(q2, self.q2_min),
                                               tf.less(q2, self.q2_max)))
        inside = tf.logical_and(inside, tf.logical_and(tf.greater(Bmass, self.Bmass_min),
                                                       tf.less(Bmass, self.Bmass_max)))
        inside = tf.logical_and(inside, tf.logical_and(tf.greater(Kpimass, self.Kpimass_min),
                                                       tf.less(Kpimass, self.Kpimass_max)))

        return inside

    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [
            np.random.uniform(-1., 1., size).astype('d'),
            np.random.uniform(-1., 1., size).astype('d'),
            np.random.uniform(-math.pi, math.pi, size).astype('d'),
            np.random.uniform(self.q2_min, self.q2_max, size).astype('d'),
            np.random.uniform(self.Bmass_min, self.Bmass_max, size).astype('d'),
            np.random.uniform(self.Kpimass_min, self.Kpimass_max, size).astype('d')
            ]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def uniform_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, majorant))

    def rectangular_grid_sample(self, size_cos_1, size_cos_2, size_phi, size_q2, size_Bmass,
                                size_Kpimass):
        """
        Create a data sample in the form of rectangular grid of points within the phase space.
        Useful for normalisation.
        """
        size = size_cos_1 * size_cos_2 * size_phi * size_q2 * size_Bmass * size_Kpimass
        mgrid = np.lib.index_tricks.nd_grid()
        v1 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 0] * 2. / float(size_cos_1) - 1.
        v2 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 1] * 2. / float(size_cos_2) - 1.
        v3 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 2] * 2. * math.pi / float(size_phi)
        v4 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 3] * (self.q2_max - self.q2_min) / float(size_q2) + self.q2_min
        v5 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 4] * \
             (self.Bmass_max - self.Bmass_min) / float(size_Bmass) + self.Bmass_min
        v6 = mgrid[0:size_cos_1, 0:size_cos_2, 0:size_phi, 0:size_q2, 0:size_Bmass, 0:size_Kpimass][
                 5] * \
             (self.Kpimass_max - self.Kpimass_min) / float(size_Kpimass) + self.Kpimass_min

        v = [v1.reshape(size).astype('d'), v2.reshape(size).astype('d'),
             v3.reshape(size).astype('d'), v4.reshape(size).astype('d'),
             v5.reshape(size).astype('d'), v6.reshape(size).astype('d')]
        x = tf.stack(v, axis=1)
        return tf.boolean_mask(x, self.inside(x))

    def cos_theta_1(self, sample):
        """
        Return cos_theta_1 variable (vector) for the input sample
        """
        return sample[:, 0]

    def cos_theta_2(self, sample):
        """
        Return cos_theta_2 variable (vector) for the input sample
        """
        return sample[:, 1]

    def phi(self, sample):
        """
        Return phi variable (vector) for the input sample
        """
        return sample[:, 2]

    def Q2(self, sample):
        """
        Return q^2 variable (vector) for the input sample
        """
        return sample[:, 3]

    def B_mass(self, sample):
        """
        Return B-Mass variable (vector) for the input sample
        """
        return sample[:, 4]

    def Kpi_mass(self, sample):
        """
        Return Kpi-Mass variable (vector) for the input sample
        """
        return sample[:, 5]

    def placeholder(self, name=None):
        return tf.placeholder(fptype, shape=(None, None), name=name)


class PHSPGenerator(object):
    def __init__(self, m_mother, m_daughters):
        """
        Constructor
        """
        self.ndaughters = len(m_daughters)
        self.m_daughters = m_daughters
        self.m_mother = m_mother

    def random_ordered(self, nev):
        return (-1) * tf.nn.top_k(-tf.random_uniform([nev, self.ndaughters - 2], dtype=tf.float64),
                                  k=self.ndaughters - 2).values

    def generate_flat_angles(self, nev):
        return (tf.random_uniform([nev], minval=-1., maxval=1., dtype=tf.float64),
                tf.random_uniform([nev], minval=-math.pi, maxval=math.pi, dtype=tf.float64))

    def rotate_lorentz_vector(self, p, costheta, phi):
        pvec = spatial_component(p)
        energy = time_component(p)
        pvecrot = self.rotate_vector(pvec, costheta, phi)
        return lorentz_vector(pvecrot, energy)

    def rotate_vector(self, vec, costheta, phi):
        cZ = costheta
        sZ = tf.sqrt(1 - cZ ** 2)
        cY = tf.cos(phi)
        sY = tf.sin(phi)
        x = x_component(vec)
        y = y_component(vec)
        z = z_component(vec)
        xnew = cZ * x - sZ * y
        ynew = sZ * x + cZ * y
        xnew2 = cY * xnew - sY * z
        znew = sY * xnew + cY * z
        return vector(xnew2, ynew, znew)

    def generate_model(self, nev):
        rands = self.random_ordered(nev)
        delta = self.m_mother - sum(self.m_daughters)

        sumsubmasses = []
        for i in range(self.ndaughters - 2):
            sumsubmasses.append(sum(self.m_daughters[:(i + 2)]))
        SubMasses = rands * delta + sumsubmasses
        SubMasses = tf.concat([SubMasses, scalar(self.m_mother * tf.ones([nev], dtype=tf.float64))],
                              axis=1)
        pout = []
        weights = tf.ones([nev], dtype=tf.float64)
        for i in range(self.ndaughters - 1):
            submass = tf.unstack(SubMasses, axis=1)[i]
            zeros = tf.zeros_like(submass)
            ones = tf.ones_like(submass)
            if i == 0:
                MassDaughterA = self.m_daughters[i] * ones
                MassDaughterB = self.m_daughters[i + 1] * ones
            else:
                MassDaughterA = tf.unstack(SubMasses, axis=1)[i - 1]
                MassDaughterB = self.m_daughters[i + 1] * tf.ones_like(MassDaughterA)
            pMag = two_body_momentum(submass, MassDaughterA, MassDaughterB)
            (costheta, phi) = self.generate_flat_angles(nev)
            vecArot = self.rotate_vector(vector(zeros, pMag, zeros), costheta, phi)
            pArot = lorentz_vector(vecArot, tf.sqrt(MassDaughterA ** 2 + pMag ** 2))
            pBrot = lorentz_vector(-vecArot, tf.sqrt(MassDaughterB ** 2 + pMag ** 2))
            pout = [lorentz_boost(p, spatial_component(pArot) / scalar(time_component(pArot))) for p
                    in pout]
            if i == 0:
                pout.append(pArot)
                pout.append(pBrot)
            else:
                pout.append(pBrot)
            weights = tf.multiply(weights, pMag)
        moms = tf.concat(pout, axis=1)
        phsp_model = tf.concat([moms, scalar(weights)], axis=1)
        return phsp_model


class NBody(object):
    """
    Class for N-body decay expressed as:
        m_mother   : mass of the mother
        m_daughs   : list of daughter masses
    """

    def __init__(self, m_mother, m_daughs):
        """
        Constructor
        """
        self.ndaughters = len(m_daughs)

        self.PHSPGenerator = PHSPGenerator(m_mother, m_daughs)
        self.nev_ph = tf.placeholder(tf.int32)
        self.majorant_ph = tf.placeholder(tf.float64)
        self.phsp_model = self.PHSPGenerator.generate_model(self.nev_ph)
        self.phsp_model_majorant = tf.concat([self.phsp_model, scalar(
            tf.random_uniform([self.nev_ph], minval=0., maxval=self.majorant_ph, dtype=fptype))],
                                             axis=1)

        self.data_placeholder = self.placeholer("data")
        self.norm_placeholder = self.placeholer("norm")

    def filter(self, x):
        return x

    def density(self, x):
        return tf.transpose(x)[4 * self.ndaughters]

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        with tf.Session() as s:
            feed_dict = {self.nev_ph: size}
            if majorant > 0:
                feed_dict.update({self.majorant_ph: majorant})
            uniform_sample = s.run(self.phsp_model_majorant if majorant > 0 else self.phsp_model,
                                   feed_dict=feed_dict)
        return uniform_sample

    def final_state_momenta(self, x):
        """
        Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
        defined by the phase space vector x. The momenta are calculated in the
        D rest frame.
        """
        p3s = [
            vector(tf.transpose(x)[4 * i], tf.transpose(x)[4 * i + 1], tf.transpose(x)[4 * i + 2])
            for i in range(self.ndaughters)]
        pLs = [lorentz_vector(p3, tf.transpose(x)[4 * i + 3]) for p3, i in
               zip(p3s, range(self.ndaughters))]
        return tuple(pL for pL in pLs)

    def placeholer(self, name=None):
        return tf.placeholder(fptype, shape=(None, None), name=name)


# class FourBody :
#   """
#   Class for 4-body decay phase space D->(A1 A2)(B1 B2) expressed as:
#     ma   : invariant mass of the A1 A2 combination
#     mb   : invariant mass of the B1 B2 combination
#     hela : cosine of the helicity angle of A1
#     helb : cosine of the helicity angle of B1
#     phi  : angle between the A1 A2 and B1 B2 planes in D rest frame
#   """
#   def __init__(self, ma1, ma2, mb1, mb2, md, useROOT=True ) :
#     """
#       Constructor
#     """
#     self.ma1 = ma1
#     self.ma2 = ma2
#     self.mb1 = mb1
#     self.mb2 = mb2
#     self.md = md
#     self.ndaughters=4
#
#     self.PHSPGenerator = PHSPGenerator(md,[ma1,ma2,mb1,mb2])
#     self.nev_ph = tf.placeholder(tf.int32)
#     self.majorant_ph = tf.placeholder(tf.float64)
#     self.phsp_model = self.PHSPGenerator.generate_model(self.nev_ph)
#     self.phsp_model_majorant = tf.concat([self.phsp_model,scalar(tf.random_uniform([self.nev_ph],
#     minval=0.,maxval=self.majorant_ph,dtype=fptype))],axis=1)
#
#     self.data_placeholder = self.placeholder("data")
#     self.norm_placeholder = self.placeholder("data")
#
#   def density(self, x) :
#     return tf.transpose(x)[4*self.ndaughters]
#
#   def uniform_sample(self, size, majorant = -1) :
#     """
#       Generate uniform sample of point within phase space.
#         size     : number of _initial_ points to generate. Not all of them will fall into phase
#         space,
#                    so the number of points in the output will be <size.
#         majorant : if majorant>0, add 3rd dimension to the generated tensor which is
#                    uniform number from 0 to majorant. Useful for accept-reject toy MC.
#       Note it does not actually generate the sample, but returns the data flow graph for
# generation,
#       which has to be run within TF session.
#     """
#     s = tf.Session()
#     feed_dict = {self.nev_ph : size}
#     if majorant>0: feed_dict.update({self.majorant_ph : majorant})
#     uniform_sample = s.run( self.phsp_model_majorant if majorant > 0 else self.phsp_model,
#                             feed_dict= feed_dict )
#     s.close()
#     return uniform_sample
#
#   def final_state_momenta(self, x) :
#     """
#        Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
#        defined by the phase space vector x. The momenta are calculated in the
#        D rest frame.
#     """
#     p3s = [ vector(tf.transpose(x)[4*i],tf.transpose(x)[4*i+1],tf.transpose(x)[4*i+2]) for i in
#     range(self.ndaughters) ]
#     pLs = [ lorentz_vector(p3,tf.transpose(x)[4*i+3]) for p3,i in zip(p3s,range(self.ndaughters)) ]
#     return tuple(pL for pL in pLs)
#
#   def placeholder(self, name = None) :
#     return tf.placeholder(fptype, shape = (None, None), name = name )


class FourBodyHelicityPhaseSpace(object):
    """
    Class for 4-body decay phase space D->(A1 A2)(B1 B2) expressed as:
        ma   : invariant mass of the A1 A2 combination
        mb   : invariant mass of the B1 B2 combination
        hela : cosine of the helicity angle of A1
        helb : cosine of the helicity angle of B1
        phi  : angle between the A1 A2 and B1 B2 planes in D rest frame
    """

    def __init__(self, ma1, ma2, mb1, mb2, md):
        """
        Constructor
        """
        self.ma1 = ma1
        self.ma2 = ma2
        self.mb1 = mb1
        self.mb2 = mb2
        self.md = md

        self.ma1a2min = self.ma1 + self.ma2
        self.ma1a2max = self.md - self.mb1 - self.mb2
        self.mb1b2min = self.mb1 + self.mb2
        self.mb1b2max = self.md - self.ma1 - self.ma2

        self.data_placeholder = self.placeholder("data")
        self.norm_placeholder = self.placeholder("norm")

    def inside(self, x):
        """
        Check if the point x is inside the phase space
        """
        ma1a2 = self.Ma1a2(x)
        mb1b2 = self.Mb1b2(x)
        ctha = self.cos_helicity_A(x)
        cthb = self.cos_helicity_B(x)
        phi = self.phi(x)

        inside = tf.logical_and(tf.logical_and(tf.greater(ctha, -1.), tf.less(ctha, 1.)),
                                tf.logical_and(tf.greater(cthb, -1.), tf.less(cthb, 1.)))
        inside = tf.logical_and(inside,
                                tf.logical_and(tf.greater(phi, -math.pi), tf.less(phi, math.pi))
                                )

        mb1b2max = self.md - ma1a2

        inside = tf.logical_and(inside, tf.logical_and(tf.greater(ma1a2, self.ma1a2min),
                                                       tf.less(ma1a2, self.ma1a2max)))
        inside = tf.logical_and(inside, tf.logical_and(tf.greater(mb1b2, self.mb1b2min),
                                                       tf.less(mb1b2, mb1b2max)))

        return inside

    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    def density(self, x):
        ma1a2 = self.Ma1a2(x)
        mb1b2 = self.Mb1b2(x)
        d1 = two_body_momentum(self.md, ma1a2, mb1b2)
        d2 = two_body_momentum(ma1a2, self.ma1, self.ma2)
        d3 = two_body_momentum(mb1b2, self.mb1, self.mb2)
        return d1 * d2 * d3 / self.md

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [np.random.uniform(self.ma1a2min, self.ma1a2max, size).astype('d'),
             np.random.uniform(self.mb1b2min, self.mb1b2max, size).astype('d'),
             np.random.uniform(-1., 1., size).astype('d'),
             np.random.uniform(-1., 1., size).astype('d'),
             np.random.uniform(-math.pi, math.pi, size).astype('d'),
             ]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def uniform_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, majorant))

    def Ma1a2(self, sample):
        """
        Return M2ab variable (vector) for the input sample
        """
        return sample[:, 0]

    def Mb1b2(self, sample):
        """
        Return M2bc variable (vector) for the input sample
        """
        return sample[:, 1]

    def cos_helicity_A(self, sample):
        """
        Return cos(helicity angle) of the A1A2 resonance
        """
        return sample[:, 2]

    def cos_helicity_B(self, sample):
        """
        Return cos(helicity angle) of the B1B2 resonance
        """
        return sample[:, 3]

    def phi(self, sample):
        """
        Return phi angle between A1A2 and B1B2 planes
        """
        return sample[:, 4]

    def final_state_momenta(self, x):
        """
        Return final state momenta p(A1), p(A2), p(B1), p(B2) for the decay
        defined by the phase space vector x. The momenta are calculated in the
        D rest frame.
        """
        ma1a2 = self.Ma1a2(x)
        mb1b2 = self.Mb1b2(x)
        ctha = self.cos_helicity_A(x)
        cthb = self.cos_helicity_B(x)
        phi = self.phi(x)

        p0 = two_body_momentum(self.md, ma1a2, mb1b2)
        pA = two_body_momentum(ma1a2, self.ma1, self.ma2)
        pB = two_body_momentum(mb1b2, self.mb1, self.mb2)

        zeros = tf.zeros_like(pA)

        p3A = rotate_vector(vector(zeros, zeros, pA), zeros, tf.acos(ctha), zeros)
        p3B = rotate_vector(vector(zeros, zeros, pB), zeros, tf.acos(cthb), phi)

        ea = tf.sqrt(p0 ** 2 + ma1a2 ** 2)
        eb = tf.sqrt(p0 ** 2 + mb1b2 ** 2)
        v0a = vector(zeros, zeros, p0 / ea)
        v0b = vector(zeros, zeros, -p0 / eb)

        p4A1 = lorentz_boost(lorentz_vector(p3A, tf.sqrt(self.ma1 ** 2 + norm(p3A) ** 2)), v0a)
        p4A2 = lorentz_boost(lorentz_vector(-p3A, tf.sqrt(self.ma2 ** 2 + norm(p3A) ** 2)), v0a)
        p4B1 = lorentz_boost(lorentz_vector(p3B, tf.sqrt(self.mb1 ** 2 + norm(p3B) ** 2)), v0b)
        p4B2 = lorentz_boost(lorentz_vector(-p3B, tf.sqrt(self.mb2 ** 2 + norm(p3B) ** 2)), v0b)

        return p4A1, p4A2, p4B1, p4B2

    def placeholder(self, name=None):
        return tf.placeholder(fptype, shape=(None, None), name=name)


class RectangularPhaseSpace(object):
    """
    Class for rectangular phase space in n dimensions
    """

    def __init__(self, ranges=(0., 1.)):
        """
        Constructor
        """
        self.data_placeholder = self.placeholder("data")
        self.norm_placeholder = self.placeholder("norm")
        self.ranges = ranges

    def inside(self, x):
        """
        Check if the point x is inside the phase space
        """
        inside = tf.constant([True], dtype=bool)
        for n, r in enumerate(self.ranges):
            var = self.coordinate(x, n)
            inside = tf.logical_and(inside,
                                    tf.logical_and(tf.greater(var, r[0]), tf.less(var, r[1])))
        return inside

    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    def unfiltered_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [np.random.uniform(r[0], r[1], size).astype('d') for r in self.ranges]
        if majorant > 0:
            v += [np.random.uniform(0., majorant, size).astype('d')]
        return np.transpose(np.array(v))

    def uniform_sample(self, size, majorant=-1):
        """
        Generate uniform sample of point within phase space.
            size     : number of _initial_ points to generate. Not all of them will fall into phase
            space,
                       so the number of points in the output will be <size.
            majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                       uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for
        generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, majorant))

    def rectangular_grid_sample(self, sizes):
        """
        Create a data sample in the form of rectangular grid of points within the phase space.
        Useful for normalisation.
        """
        size = 1
        for i in sizes:
            size *= i
        v = []
        mg = np.mgrid[[slice(0, i) for i in sizes]]
        for i, (r, s) in enumerate(zip(self.ranges, sizes)):
            v1 = (mg[i] + 0.5) * (r[1] - r[0]) / float(s) + r[0]
            v += [v1.reshape(size).astype('d')]
        x = tf.stack(v, axis=1)
        return tf.boolean_mask(x, self.inside(x))

    def coordinate(self, sample, n):
        """
        Return coordinate number n from the input sample
        """
        return sample[:, n]

    def placeholder(self, name=None):
        return tf.placeholder(fptype, shape=(None, None), name=name)

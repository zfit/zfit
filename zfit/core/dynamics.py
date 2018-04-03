from __future__ import print_function, division, absolute_import

import tensorflow as tf

import zfit
from zfit.core.kinematics import two_body_momentum, ComplexTwoBodyMomentum
from zfit.settings import ctype
from . import tfext


def helicity_amplitude(x, spin):
    """
    Helicity amplitude for a resonance in scalar-scalar state
      x    : cos(helicity angle)
      spin : spin of the resonance
    """
    if spin == 0:
        return tf.complex(tfext.constant(1.), tfext.constant(0.))
    elif spin == 1:
        return tf.complex(x, tfext.constant(0.))
    elif spin == 2:
        return tf.complex((3. * x ** 2 - 1.) / 2., tfext.constant(0.))
    elif spin == 3:
        return tf.complex((5. * x ** 3 - 3. * x) / 2., tfext.constant(0.))
    elif spin == 4:
        return tf.complex((35. * x ** 4 - 30. * x ** 2 + 3.) / 8., tfext.constant(0.))
    else:
        raise ValueError("Illegal spin number.")


def relativistic_breit_wigner(m2, mres, wres):
    """
    Relativistic Breit-Wigner
    """
    if wres.dtype is ctype:
        return 1. / (tfext.to_complex(mres ** 2 - m2) - tf.complex(tfext.constant(0.), mres) * wres)
    if wres.dtype is zfit.settings.fptype:
        return 1. / tf.complex(mres ** 2 - m2, -mres * wres)
    return None


def blatt_weisskopf_ff(q, q0, d, l):
    """
    Blatt-Weisskopf formfactor for intermediate resonance
    """
    z = q * d
    z0 = q0 * d

    def hankel1(x):
        if l == 0:
            return tfext.constant(1.)
        if l == 1:
            return 1 + x ** 2
        if l == 2:
            x2 = x ** 2
            return 9 + x2 * (3. + x2)
        if l == 3:
            x2 = x ** 2
            return 225 + x2 * (45 + x2 * (6 + x2))
        if l == 4:
            x2 = x ** 2
            return 11025. + x2 * (1575. + x2 * (135. + x2 * (10. + x2)))

    return tf.sqrt(hankel1(z0) / hankel1(z))


def mass_dependent_width(m, m0, gamma0, p, p0, ff, l):
    """
    Mass-dependent width for BW amplitude
    """
    return gamma0 * ((p / p0) ** (2 * l + 1)) * (m0 / m) * (ff ** 2)


def orbital_barrier_factor(p, p0, l):
    """
    Orbital barrier factor
    """
    return (p / p0) ** l


def breit_wigner_line_shape(m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrier_factor=True,
                            ma0=None, md0=None):
    """
    Breit-Wigner amplitude with Blatt-Weisskopf formfactors, mass-dependent width and orbital
    barriers
    """
    m = tf.sqrt(m2)
    q = two_body_momentum(md, m, mc)
    q0 = two_body_momentum(md if md0 is None else md0, m0, mc)
    p = two_body_momentum(m, ma, mb)
    p0 = two_body_momentum(m0, ma if ma0 is None else ma0, mb)
    ffr = blatt_weisskopf_ff(p, p0, dr, lr)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    width = mass_dependent_width(m, m0, gamma0, p, p0, ffr, lr)
    bw = relativistic_breit_wigner(m2, m0, width)
    ff = ffr * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * tf.complex(ff, tfext.constant(0.))


def subthreshold_breit_wigner_line_shape(m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld,
                                         barrier_factor=True):
    """
    Breit-Wigner amplitude (with the mass under kinematic threshold)
    with Blatt-Weisskopf formfactors, mass-dependent width and orbital barriers
    """
    m = tf.sqrt(m2)
    mmin = ma + mb
    mmax = md - mc
    tanhterm = tf.tanh((m0 - ((mmin + mmax) / 2.)) / (mmax - mmin))
    m0eff = mmin + (mmax - mmin) * (1. + tanhterm) / 2.
    q = two_body_momentum(md, m, mc)
    q0 = two_body_momentum(md, m0eff, mc)
    p = two_body_momentum(m, ma, mb)
    p0 = two_body_momentum(m0eff, ma, mb)
    ffr = blatt_weisskopf_ff(p, p0, dr, lr)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    width = mass_dependent_width(m, m0, gamma0, p, p0, ffr, lr)
    bw = relativistic_breit_wigner(m2, m0, width)
    ff = ffr * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * tf.complex(ff, tfext.constant(0.))


def exp_non_resonant_line_shape(m2, m0, alpha, ma, mb, mc, md, lr, ld, barrier_factor=True):
    """
    Exponential nonresonant amplitude with orbital barriers
    """
    if barrier_factor:
        m = tf.sqrt(m2)
        q = two_body_momentum(md, m, mc)
        q0 = two_body_momentum(md, m0, mc)
        p = two_body_momentum(m, ma, mb)
        p0 = two_body_momentum(m0, ma, mb)
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        return tf.complex(b1 * b2 * tf.exp(-alpha * (m2 - m0 ** 2)), tfext.constant(0.))
    else:
        return tf.complex(tf.exp(-alpha * (m2 - m0 ** 2)), tfext.constant(0.))


def gounaris_sakurai_line_shape(s, m, gamma, m_pi):
    """
      Gounaris-Sakurai shape for rho->pipi
        s     : squared pipi inv. mass
        m     : rho mass
        gamma : rho width
        m_pi  : pion mass
    """
    m2 = m * m
    m_pi2 = m_pi * m_pi
    ss = tf.sqrt(s)

    ppi2 = (s - 4. * m_pi2) / 4.
    p02 = (m2 - 4. * m_pi2) / 4.
    p0 = tf.sqrt(p02)
    ppi = tf.sqrt(ppi2)

    hs = 2. * ppi / tfext.pi / ss * tf.log((ss + 2. * ppi) / 2. / m_pi)
    hm = 2. * p0 / tfext.pi / m * tf.log((m + 2. * ppi) / 2. / m_pi)

    dhdq = hm * (1. / 8. / p02 - 1. / 2. / m2) + 1. / 2. / tfext.pi / m2
    f = gamma * m2 / (p0 ** 3) * (ppi2 * (hs - hm) - p02 * (s - m2) * dhdq)

    gamma_s = gamma * m2 * (ppi ** 3) / s / (p0 ** 3)

    dr = m2 - s + f
    di = ss * gamma_s

    r = dr / (dr ** 2 + di ** 2)
    i = di / (dr ** 2 + di ** 2)

    return tf.complex(r, i)


def flatten_line_shape(s, m, g1, g2, ma1, mb1, ma2, mb2):
    """
      Flatte line shape
        s : squared inv. mass
        m : resonance mass
        g1 : coupling to ma1, mb1
        g2 : coupling to ma2, mb2
    """
    mab = tf.sqrt(s)
    pab1 = two_body_momentum(mab, ma1, mb1)
    rho1 = 2. * pab1 / mab
    pab2 = two_body_momentum(mab, ma2, mb2, calc_complex=True)
    rho2 = 2. * pab2 / tfext.to_complex(mab)
    gamma = ((tfext.to_complex(g1 ** 2 * rho1) + tfext.to_complex(g2 ** 2) *
              rho2) / tfext.to_complex(m))
    return relativistic_breit_wigner(s, m, gamma)


# New! Follow the definition of (not completely)LHCb-PAPER-2016-141, and Rafa's thesis(Laura++)
def LASS_line_shape(m2ab, m0, gamma0, a, r, ma, mb, dr, lr, barrier_factor=True):
    """
    LASS amplitude with Blatt-Weisskopf formfactors, mass-dependent width and orbital barriers
    """
    m = tf.sqrt(m2ab)
    q = two_body_momentum(m, ma, mb)
    q0 = two_body_momentum(m0, ma, mb)
    ffr = blatt_weisskopf_ff(q, q0, dr, lr)
    non_res_LASS = non_resonant_LASS_line_shape(m2ab, a, r, ma, mb)
    res_LASS = resonant_LASS_line_shape(m2ab, m0, gamma0, a, r, ma, mb)
    ff = ffr
    if barrier_factor:
        b1 = orbital_barrier_factor(q, q0, lr)
        ff *= b1
    return tf.complex(ff, tfext.constant(0.)) * (non_res_LASS + res_LASS)


def non_resonant_LASS_line_shape(m2ab, a, r, ma, mb):
    """
      LASS line shape, nonresonant part
    """
    m = tf.sqrt(m2ab)
    q = two_body_momentum(m, ma, mb)
    cot_deltab = 1. / a / q + 1. / 2. * r * q
    ampl = tfext.to_complex(m) / tf.complex(q * cot_deltab, -q)
    return ampl


# Modified to match the definition of LHCb-PAPER-2016-141 (and also Laura++)
def resonant_LASS_line_shape(m2ab, m0, gamma0, a, r, ma, mb):
    """
      LASS line shape, resonant part
    """
    m = tf.sqrt(m2ab)
    q0 = two_body_momentum(m0, ma, mb)
    q = two_body_momentum(m, ma, mb)
    cot_deltab = 1. / a / q + 1. / 2. * r * q
    phase = tf.atan(1. / cot_deltab)
    width = gamma0 * q / m * m0 / q0
    ampl = (relativistic_breit_wigner(m2ab, m0, width) *
            tf.complex(tf.cos(2. * phase), tf.sin(2. * phase)) *
            tfext.to_complex(m0 * m0 * gamma0 / q0))
    return ampl


def dabba_line_shape(m2ab, b, alpha, beta, ma, mb):
    """
      Dabba line shape
    """
    m_sum = ma + mb
    m2a = ma ** 2
    m2b = mb ** 2
    s_adler = max(m2a, m2b) - 0.5 * min(m2a, m2b)
    m_sum2 = m_sum * m_sum
    m_diff = m2ab - m_sum2
    rho = tf.sqrt(1. - m_sum2 / m2ab)
    real_part = 1.0 - beta * m_diff
    imag_part = b * tf.exp(-alpha * m_diff) * (m2ab - s_adler) * rho
    denom_factor = real_part * real_part + imag_part * imag_part
    ampl = tf.complex(real_part, imag_part) / tfext.to_complex(denom_factor)
    return ampl

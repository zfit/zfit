import tensorflow as tf
import numpy as np
import itertools

# Use double precision throughout
fptype = tf.float64

# Use double precision throughout
ctype = tf.complex128


# Sum of the list components

# Density for a complex amplitude
def Density(ampl): return tf.abs(ampl) ** 2


# Create a complex number from a magnitude and a phase
def Polar(a, ph): return tf.complex(a * tf.cos(ph), a * tf.sin(ph))


# Cast a real number to complex
def CastComplex(re): return tf.cast(re, dtype=ctype)


# Declare constant
def Const(c): return tf.constant(c, dtype=fptype)


# Declare invariant
def Invariant(c): return tf.constant([c], dtype=fptype)


# |x|^2
def AbsSq(x): return tf.real(x * tf.conj(x))


# Pi
def Pi(): return Const(np.pi)


# Return argument of a complex number
def Argument(c): return tf.atan2(tf.imag(c), tf.real(c))


def Clebsch(j1, m1, j2, m2, J, M):
    """
      Return Clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
      (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
    """
    from sympy.physics.quantum.cg import CG
    from sympy import Rational
    return CG(Rational(j1, 2), Rational(m1, 2), Rational(j2, 2), Rational(m2, 2), Rational(J, 2),
              Rational(M, 2)).doit().evalf()


def Interpolate(t, c):
    """
      Multilinear interpolation on a rectangular grid of arbitrary number of dimensions
        t : TF tensor representing the grid (of rank N)
        c : Tensor of coordinates for which the interpolation is performed
        return: 1D tensor of interpolated values
    """
    rank = len(t.get_shape())
    ind = tf.cast(tf.floor(c), tf.int32)
    t2 = tf.pad(t, rank * [[1, 1]], 'SYMMETRIC')
    wts = []
    for vertex in itertools.product([0, 1], repeat=rank):
        ind2 = ind + tf.constant(vertex, dtype=tf.int32)
        weight = tf.reduce_prod(1. - tf.abs(c - tf.cast(ind2, dtype=fptype)), 1)
        wt = tf.gather_nd(t2, ind2 + 1)
        wts += [weight * wt]
    interp = tf.reduce_sum(tf.stack(wts), 0)
    return interp


def SetSeed(seed):
    """
      Set random seed for numpy
    """
    np.random.seed(seed)

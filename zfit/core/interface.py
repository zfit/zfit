import tensorflow as tf
import numpy as np
import itertools

# Use double precision throughout
fptype = tf.float64

# Use double precision throughout
ctype = tf.complex128

# Sum of the list components
def Sum( ampls ) : return tf.add_n( ampls )

# Density for a complex amplitude
def Density(ampl) : return tf.abs(ampl)**2

# Absolute value
def Abs(a) : return tf.abs(a)

# Create a complex number from real and imaginary parts
def Complex(re, im) : return tf.complex(re, im)

# Create a complex number from a magnitude and a phase
def Polar(a, ph) : return tf.complex(a*tf.cos(ph), a*tf.sin(ph))

# Cast a real number to complex
def CastComplex(re) : return tf.cast(re, dtype = ctype)

# Return complex conjugate
def Conj(c) : return tf.conj(c)

# Return the real part of a complex number
def Real(c): return tf.real(c)

# Return the imaginary part of a complex number
def Im(c): return tf.imag(c)

# Declare constant
def Const(c) : return tf.constant(c, dtype=fptype)

# Declare invariant
def Invariant(c) : return tf.constant([c], dtype=fptype)

# |x|^2
def AbsSq(x) : return Real(x * tf.conj(x))

# Square
def Square(x) : return tf.square(x)

# Sqrt
def Sqrt(x) : return tf.sqrt(x)

# Exp
def Exp(x) : return tf.exp(x)

# Log
def Log(x) : return tf.log(x)

# Sin
def Sin(x) : return tf.sin(x)

# Cos
def Cos(x) : return tf.cos(x)

# Asin
def Asin(x) : return tf.asin(x)

# Atan
def Atan(x) : return tf.atan(x)

# Acos
def Acos(x) : return tf.acos(x)

# Tanh
def Tanh(x) : return tf.tanh(x)

# Pow
def Pow(x,p): return tf.pow(x,p)

# Pi
def Pi() : return Const( np.pi )

# Create a tensor with zeros of the same shape as input tensor
def Zeros(x) : return tf.zeros_like(x)

# Create a tensor with ones of the same shape as input tensor
def Ones(x) : return tf.ones_like(x)

# Atan2 was not defined in TensorFlow, used own implementation
# It exists since v1.2
def Atan2(y, x):
  return tf.atan2(y, x)
#  tolerance = 1e-14
#  angle = tf.where(tf.greater(x,tolerance), tf.atan(y/x), tf.zeros_like(x))
#  angle = tf.where(tf.logical_and(tf.less(x,tolerance),  tf.greater_equal(y,-tolerance)), tf.atan(y/x) + Pi(), angle)
#  angle = tf.where(tf.logical_and(tf.less(x,tolerance),  tf.less(y,tolerance)), tf.atan(y/x) - Pi(), angle)
#  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*Pi() * tf.ones_like(x), angle)
#  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*Pi() * tf.ones_like(x), angle)
#  angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
#  return angle

# Return argument of a compelx number
def Argument(c) : return Atan2(tf.imag(c), tf.real(c))

def Clebsch(j1, m1, j2, m2, J, M) : 
  """
    Return Clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
    (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
  """
  from sympy.physics.quantum.cg import CG
  from sympy import Rational
  return CG(Rational(j1,2), Rational(m1,2), Rational(j2,2), Rational(m2,2), Rational(J,2), Rational(M,2) ).doit().evalf()

def Interpolate(t, c) :
  """
    Multilinear interpolation on a rectangular grid of arbitrary number of dimensions
      t : TF tensor representing the grid (of rank N)
      c : Tensor of coordinates for which the interpolation is performed
      return: 1D tensor of interpolated values
  """
  rank = len(t.get_shape())
  ind = tf.cast(tf.floor(c), tf.int32)
  t2 = tf.pad(t, rank*[[1,1]], 'SYMMETRIC')
  wts = []
  for vertex in itertools.product([0, 1], repeat = rank) :
    ind2 = ind + tf.constant(vertex, dtype = tf.int32)
    weight = tf.reduce_prod(1. - tf.abs(c - tf.cast(ind2, dtype = fptype)), 1)
    wt = tf.gather_nd(t2, ind2+1)
    wts += [ weight*wt ]
  interp = tf.reduce_sum(tf.stack(wts), 0)
  return interp

def SetSeed(seed) :
  """
    Set random seed for numpy
  """
  np.random.seed(seed)

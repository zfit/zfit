from __future__ import print_function, division, absolute_import

import tensorflow as tf

from zfit.settings import ctype, fptype


# Density for a complex amplitude
def Density(ampl): return tf.abs(ampl) ** 2


# Create a complex number from a magnitude and a phase
def Polar(a, ph):
    """Create a complex number from magnitude and phase"""
    return tf.complex(a * tf.cos(ph), a * tf.sin(ph))


# Cast a real number to complex
def CastComplex(re): return tf.cast(re, dtype=ctype)


# Declare constant
def Const(c): return tf.constant(c, dtype=fptype)


# Declare invariant
def Invariant(c): return tf.constant([c], dtype=fptype)


# |x|^2
def AbsSq(x): return tf.real(x * tf.conj(x))


def Argument(c): return tf.atan2(tf.imag(c), tf.real(c))

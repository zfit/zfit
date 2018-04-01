from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

import zfit.settings


# Density for a complex amplitude
def Density(ampl): return tf.abs(ampl) ** 2


# Create a complex number from a magnitude and a phase
def Polar(a, ph):
    """Create a complex number from magnitude and phase"""
    return tf.complex(a * tf.cos(ph), a * tf.sin(ph))


def nth_pow(x, n, name=None):
    """Calculate the nth power of the complex Tensor x.

    Args:
        x (tf.Tensor, complex):
        n (int >= 0): Power
        name (str): No effect, for API compatibility with tf.pow
    """
    if not n >= 0:
        raise ValueError("n (power) has to be >= 0. Currently, n={}".format(n))

    power = 1
    for _ in range(n):
        power *= x
    return power


# Cast a real number to complex
def to_complex(number): return tf.cast(number, dtype=zfit.settings.ctype)


def to_real(number):
    return tf.cast(number, dtype=zfit.settings.fptype)


# Declare constant
def constant(c): return tf.constant(c, dtype=zfit.settings.fptype)


# Declare invariant
def Invariant(c): return tf.constant([c], dtype=zfit.settings.fptype)


# |x|^2
def AbsSq(x): return tf.real(x * tf.conj(x))


def Argument(c): return tf.atan2(tf.imag(c), tf.real(c))


pi = constant(np.pi)

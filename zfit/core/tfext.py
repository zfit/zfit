from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

import zfit.settings
import zfit.core.math as zmath

from zfit.settings import types as ztypes

pi = tf.constant(zmath.pi, dtype=ztypes.float)
inf = tf.constant(zmath.inf, dtype=ztypes.float)


# density for a complex amplitude
def density(ampl): return tf.abs(ampl) ** 2


# Create a complex number from a magnitude and a phase
def polar(a, ph):
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

    power = to_complex(1.)
    for _ in range(n):
        power *= x
    return power


# Cast a real number to complex
def to_complex(number): return tf.cast(number, dtype=ztypes.complex)


def to_real(number):
    return tf.cast(number, dtype=ztypes.float)


# Declare constant
def constant(c): return tf.constant(c, dtype=ztypes.float)


# Declare invariant
def invariant(c): return tf.constant([c], dtype=ztypes.float)


# |x|^2
def abs_square(x): return tf.real(x * tf.conj(x))


def argument(c): return tf.atan2(tf.imag(c), tf.real(c))


pi = constant(np.pi)

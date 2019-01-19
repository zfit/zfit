import tensorflow as tf

from zfit import ztf
import math as _mt
from ..settings import ztypes

ztf.pi = tf.constant(_mt.pi, dtype=ztypes.float)


# density for a complex amplitude
def density(ampl): return tf.abs(ampl) ** 2


# Create a complex number from a magnitude and a phase
def polar(a, ph):
    """Create a complex number from magnitude and phase"""
    return tf.complex(a * tf.cos(ph), a * tf.sin(ph))


def invariant(c): return tf.constant([c], dtype=ztypes.float)


# |x|^2


def argument(c): return tf.atan2(tf.imag(c), tf.real(c))

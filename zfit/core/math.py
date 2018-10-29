from __future__ import print_function, division, absolute_import

import itertools

import tensorflow as tf

from zfit.settings import types as ztypes
from zfit.core import tfext
from zfit import ztf

def poly_complex(*args, real_x=False):  # py23 compatibility: change **kwargs to real_x=False
    """Complex polynomial with the last arg being x.

    Args:
        *args (tf.Tensor or equ.): Coefficients of the polynomial
        real_x (bool): If True, x is assumed to be real.

    Returns:
        tf.Tensor:
    """

    args = list(args)
    x = args.pop()
    if real_x is not None:
        pow_func = tf.pow
    else:
        pow_func = ztf.nth_pow
    return tf.add_n([coef * ztf.to_complex(pow_func(x, p)) for p, coef in enumerate(args)])


def interpolate(t, c):
    """Multilinear interpolation on a rectangular grid of arbitrary number of dimensions.

    Args:
        t (tf.Tensor): Grid (of rank N)
        c (tf.Tensor): Tensor of coordinates for which the interpolation is performed

    Returns:
        tf.Tensor: 1D tensor of interpolated values
    """
    rank = len(t.get_shape())
    ind = tf.cast(tf.floor(c), tf.int32)
    t2 = tf.pad(t, rank * [[1, 1]], 'SYMMETRIC')
    wts = []
    for vertex in itertools.product([0, 1], repeat=rank):
        ind2 = ind + tf.constant(vertex, dtype=tf.int32)
        weight = tf.reduce_prod(1. - tf.abs(c - tf.cast(ind2, dtype=ztypes.float)), 1)
        wt = tf.gather_nd(t2, ind2 + 1)
        wts += [weight * wt]
    interp = tf.reduce_sum(tf.stack(wts), 0)
    return interp


def gradient_par(func):
    """Return TF graph for analytic gradient_par of the input func wrt all floating variables.

    Arguments:
            func (Tensor): A function of which the derivatives with respect to the free floating
                           parameters will be taken.
    Return:
        list(graph): the derivative
    """
    tfpars = tf.trainable_variables()  # Create TF variables
    float_tfpars = [p for p in tfpars if p.floating()]  # List of floating parameters
    return tf.gradients(func, float_tfpars)  # Get analytic gradient_par

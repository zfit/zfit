from __future__ import print_function, division, absolute_import

import tensorflow as tf
import array
import numpy as np
import math

from .interface import *
from .optimization import *


def MultivariateGauss(x, norm, mean, invCov):
    print()
    norm
    dx = x - mean
    expArg = tf.einsum("ai,ij,aj->a", dx, invCov, dx)
    return (norm ** 2) * tf.exp(-0.5 * expArg)


def Gauss2D(x, norm, xmean, ymean, xsigma, ysigma, corr):
    print()
    norm
    offdiag = abs(xsigma * ysigma) * corr
    array = [[xsigma ** 2, offdiag], [offdiag, ysigma ** 2]]
    cov = tf.stack(array)
    mean = tf.stack([xmean, ymean])
    invcov = tf.matrix_inverse(cov)
    return MultivariateGauss(x, norm, mean, invcov)


def Gauss4D(x, params):
    norm = params[0]
    mean = tf.stack(params[1:5])
    sigma = tf.stack(params[5:9])
    corr = tf.stack([[Const(1.), params[9], params[10], params[11]],
                     [params[9], Const(1.), params[12], params[13]],
                     [params[10], params[12], Const(1.), params[14]],
                     [params[11], params[13], params[14], Const(1.)]])

    cov = tf.einsum("i,ij,j->ij", sigma, corr, sigma)
    invcov = tf.matrix_inverse(cov)
    return MultivariateGauss(x, norm, mean, invcov)


class GaussianMixture2D(object):
    def __init__(self, prefix, n, x_range, y_range):
        self.params = []
        for i in range(n):
            norm = FitParameter(prefix + "n%d" % i, 1. / (1. + float(i)), 0., 2.)
            xmean = FitParameter(prefix + "xm%d" % i,
                                 np.random.uniform(x_range[0], x_range[1], 1)[0], -1., 1.)
            ymean = FitParameter(prefix + "ym%d" % i,
                                 np.random.uniform(y_range[0], y_range[1], 1)[0], -1., 1.)
            xsigma = FitParameter(prefix + "xs%d" % i, (x_range[1] - x_range[0]) / 4., 0., 2.)
            ysigma = FitParameter(prefix + "ys%d" % i, (x_range[1] - x_range[0]) / 4., 0., 2.)
            corr = FitParameter(prefix + "c%d" % i, 0., -0.9, 0.9)
            self.params += [(norm, xmean, ymean, xsigma, ysigma, corr)]
        self.params[0][0].step_size = 0.  # Fix first normalisation term

    def model(self, x):
        d = Const(0.)
        for i in self.params:
            d += Gauss2D(x, i[0], i[1], i[2], i[3], i[4], i[5])
        return d


class GaussianMixture4D(object):
    def __init__(self, prefix, n, ranges):
        self.params = []
        for i in range(n):
            norm = FitParameter(prefix + "n%d" % i, 1. / (1. + float(i)), 0., 2.)
            xmean = FitParameter(prefix + "xm%d" % i,
                                 np.random.uniform(x_range[0], x_range[1], 1)[0], -1., 1.)
            ymean = FitParameter(prefix + "ym%d" % i,
                                 np.random.uniform(y_range[0], y_range[1], 1)[0], -1., 1.)
            xsigma = FitParameter(prefix + "xs%d" % i, (x_range[1] - x_range[0]) / 4., 0., 2.)
            ysigma = FitParameter(prefix + "ys%d" % i, (x_range[1] - x_range[0]) / 4., 0., 2.)
            corr = FitParameter(prefix + "c%d" % i, 0., -0.9, 0.9)
            self.params += [(norm, xmean, ymean, xsigma, ysigma, corr)]
        self.params[0][0].step_size = 0.  # Fix first normalisation term

    def model(self, x):
        d = Const(0.)
        for i in self.params:
            d += Gauss2D(x, i[0], i[1], i[2], i[3], i[4], i[5])
        return d

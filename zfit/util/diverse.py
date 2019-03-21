import numpy as np
import tensorflow as tf

from zfit import ztf
from ..core.parameter import Parameter


def multivariate_gauss(x, norm, mean, inv_cov):
    print(norm)

    dx = x - mean
    exp_arg = tf.einsum("ai,ij,aj->a", dx, inv_cov, dx)
    return (norm ** 2) * tf.exp(-0.5 * exp_arg)


def gauss_2d(x, norm, xmean, ymean, xsigma, ysigma, corr):
    print(norm)

    offdiag = abs(xsigma * ysigma) * corr
    array = [[xsigma ** 2, offdiag], [offdiag, ysigma ** 2]]
    cov = tf.stack(array)
    mean = tf.stack([xmean, ymean])
    invcov = tf.matrix_inverse(cov)
    return multivariate_gauss(x, norm, mean, invcov)


def gauss_4d(x, params):
    norm = params[0]
    mean = tf.stack(params[1:5])
    sigma = tf.stack(params[5:9])
    corr = tf.stack([[ztf.constant(1.), params[9], params[10], params[11]],
                     [params[9], ztf.constant(1.), params[12], params[13]],
                     [params[10], params[12], ztf.constant(1.), params[14]],
                     [params[11], params[13], params[14], ztf.constant(1.)]])

    cov = tf.einsum("i,ij,j->ij", sigma, corr, sigma)
    invcov = tf.matrix_inverse(cov)
    return multivariate_gauss(x, norm, mean, invcov)


class GaussianMixture2D(object):
    def __init__(self, prefix, n, x_range, y_range):
        self.params = []
        for i in range(n):
            norm = Parameter(prefix + "n{:d}".format(i), 1. / (1. + float(i)), 0., 2.)
            xmean = Parameter(prefix + "xm{:d}".format(i),
                              np.random.uniform(x_range[0], x_range[1], 1)[0], -1., 1.)
            ymean = Parameter(prefix + "ym{:d}".format(i),
                              np.random.uniform(y_range[0], y_range[1], 1)[0], -1., 1.)
            xsigma = Parameter(prefix + "xs{:d}".format(i),
                               (x_range[1] - x_range[0]) / 4., 0., 2.)
            ysigma = Parameter(prefix + "ys{:d}".format(i),
                               (x_range[1] - x_range[0]) / 4., 0., 2.)
            corr = Parameter(prefix + "c{:d}".format(i), 0., -0.9, 0.9)
            self.params += [(norm, xmean, ymean, xsigma, ysigma, corr)]
        self.params[0][0].step_size = 0.  # Fix first normalisation term

    def model(self, x):
        d = ztf.constant(0.)
        for i in self.params:
            d += gauss_2d(x, i[0], i[1], i[2], i[3], i[4], i[5])
        return d


class GaussianMixture4D(object):
    def __init__(self, prefix, n, ranges):  # TODO: ranges? x_range, y_range?
        self.params = []
        for i in range(n):
            norm = Parameter(prefix + "n{:d}".format(i), 1. / (1. + float(i)), 0., 2.)
            xmean = Parameter(prefix + "xm{:d}".format(i),
                              np.random.uniform(x_range[0], x_range[1], 1)[0], -1., 1.)
            ymean = Parameter(prefix + "ym{:d}".format(i),
                              np.random.uniform(y_range[0], y_range[1], 1)[0], -1., 1.)
            xsigma = Parameter(prefix + "xs{:d}".format(i),
                               (x_range[1] - x_range[0]) / 4., 0., 2.)
            ysigma = Parameter(prefix + "ys{:d}".format(i),
                               (x_range[1] - x_range[0]) / 4., 0., 2.)
            corr = Parameter(prefix + "c{:d}".format(i), 0., -0.9, 0.9)
            self.params += [(norm, xmean, ymean, xsigma, ysigma, corr)]
        self.params[0][0].step_size = 0.  # Fix first normalisation term

    def model(self, x):
        d = ztf.constant(0.)
        for i in self.params:
            d += gauss_2d(x, i[0], i[1], i[2], i[3], i[4], i[5])
        return d

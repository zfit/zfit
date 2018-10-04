from __future__ import print_function, division, absolute_import

import tensorflow as tf

import zfit
from zfit.core.basepdf import BasePDF
from zfit.core.parameter import FitParameter
import zfit.core.tfext as tfz


class Gauss(BasePDF):

    def __init__(self, mu, sigma, name="Gauss"):
        super(Gauss, self).__init__(name=name, mu=mu, sigma=sigma)

    def _func(self, value):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- (value - mu) ** 2 / (tfz.constant(2.) * (sigma ** 2)))

        return gauss


class SumPDF(BasePDF):

    def __init__(self, pdfs, name="SumPDF"):
        super(SumPDF, self).__init__(name=name)
        # TODO: check if combination is possible
        self.pdfs = pdfs
        self.n_pdfs = len(self.pdfs)

    @property
    def pdfs(self):
        return self._pdfs

    @pdfs.setter
    def pdfs(self, pdfs):
        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]
        self._pdfs = pdfs

    def _func(self, value):
        # TODO: deal with yields
        func = sum([pdf.func(value) for pdf in self.pdfs])
        return func


if __name__ == '__main__':

    import numpy as np

    def true_gaussian_sum(x):
        sum_gauss = np.exp(- (x - 1.) ** 2 / (2*11. ** 2))
        sum_gauss += np.exp(- (x - 2.) ** 2 / (2*22. ** 2))
        sum_gauss += np.exp(- (x - 3.) ** 2 / (2*33. ** 2))
        return sum_gauss

    with tf.Session() as sess:
        mu1 = FitParameter("mu1", 1.)
        mu2 = FitParameter("mu2", 2.)
        mu3 = FitParameter("mu3", 3.)
        sigma1 = FitParameter("sigma1", 11.)
        sigma2 = FitParameter("sigma2", 22.)
        sigma3 = FitParameter("sigma3", 33.)

        gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss1")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss2")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss3")

        sum_gauss = SumPDF(pdfs=[gauss1, gauss2, gauss3])

        sum_gauss.norm_range = -55., 55.

        init = tf.global_variables_initializer()
        sess.run(init)


        def test_func():
            test_values = np.array([3., 129., -0.2, -78.2])
            vals = sum_gauss.func(
                tf.convert_to_tensor(test_values, dtype=zfit.settings.fptype))
            vals = sess.run(vals)
            test_sum = sum([g.func(test_values) for g in [gauss1, gauss2, gauss3]])
            print(sess.run(test_sum))
            np.testing.assert_almost_equal(vals, true_gaussian_sum(test_values))


        def test_normalization():

            low, high = sum_gauss.norm_range
            samples = tf.cast(np.random.uniform(low=low, high=high, size=100000),
                              dtype=tf.float64)
            samples.limits = low, high
            probs = sum_gauss.prob(samples)
            result = sess.run(probs)
            result = np.average(result) * (high - low)
            print(result)
            assert 0.95 < result < 1.05

        test_func()
        test_normalization()


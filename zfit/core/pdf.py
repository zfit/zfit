from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import zfit
from zfit.core.basepdf import BasePDF, WrapDistribution
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


class Normal(WrapDistribution):
    def __init__(self, loc, scale, name="Normal"):
        distribution = tfp.distributions.Normal(loc=loc, scale=scale, name=name+"_tf")
        super(Normal, self).__init__(distribution=distribution, name=name)


class SumPDF(BasePDF):

    def __init__(self, pdfs, frac=None, name="SumPDF"):
        """

        Args:
            pdfs (zfit pdf): The pdfs to multiply with each other
            frac (iterable): coefficients for the multiplication. If a scale is set, None
                will take the scale automatically. If the scale is not set, then:

                  - len(frac) = len(pdfs) - 1 results in the interpretation of a pdf. The last
                    coefficient equals to 1 - sum(frac)
                  - len(frac) = len(pdfs) results in the interpretation of multiplying each
                    function with this value
            name (str):
        """
        # Check user input, improve TODO

        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]

        # check fraction  # TODO make more flexible, allow for Tensors and unstack
        if not len(frac) in (len(pdfs), len(pdfs) - 1):
            raise ValueError("user error")  # TODO user error?

        if len(frac) == len(pdfs) - 1:
            frac = list(frac) + [1 - sum(frac)]

        super(SumPDF, self).__init__(pdfs=pdfs, frac=frac, name=name)

    def _func(self, value):
        # TODO: deal with yields
        pdfs = self.parameters['pdfs']
        frac = self.parameters['frac']
        func = tf.accumulate_n([scale * pdf.func(value) for pdf, scale in zip(pdfs, frac)])
        return func


class ProductPDF(BasePDF):
    def __init__(self, pdfs, name="ProductPDF"):
        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]
        super(ProductPDF, self).__init__(pdfs=pdfs, name=name)

    def _func(self, value):
        return np.prod([pdf.func(value) for pdf in self.parameters['pdfs']], axis=0)


if __name__ == '__main__':

    import numpy as np


    def true_gaussian_sum(x):
        sum_gauss = np.exp(- (x - 1.) ** 2 / (2 * 11. ** 2))
        sum_gauss += np.exp(- (x - 2.) ** 2 / (2 * 22. ** 2))
        sum_gauss += np.exp(- (x - 3.) ** 2 / (2 * 33. ** 2))
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


        def test_func_sum():
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


        test_func_sum()
        test_normalization()

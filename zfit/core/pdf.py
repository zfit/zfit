from __future__ import print_function, division, absolute_import

import math as mt

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from zfit.utils.exception import ExtendedPDFError
from .basepdf import BasePDF, WrapDistribution
from .parameter import FitParameter
from . import tfext as ztf
from . import math as zmath
from ..settings import types as ztypes


class Gauss(BasePDF):

    def __init__(self, mu, sigma, name="Gauss"):
        super(Gauss, self).__init__(name=name, mu=mu, sigma=sigma)

    def _unnormalized_prob(self, x):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        gauss = tf.exp(- (x - mu) ** 2 / (ztf.constant(2.) * (sigma ** 2)))

        return gauss


def _gauss_integral_from_inf_to_inf(limits, params):
    return tf.sqrt(ztf.pi * params['sigma'])


Gauss.register_analytic_integral(func=_gauss_integral_from_inf_to_inf, dims=(0,),
                                 limits=(-zmath.inf, zmath.inf))


class Normal(WrapDistribution):
    def __init__(self, loc, scale, name="Normal"):
        distribution = tfp.distributions.Normal(loc=loc, scale=scale, name=name + "_tf")
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

        # check if all are extended or None is extended
        extended_pdfs = [pdf for pdf in pdfs if pdf.is_extended]
        all_extended = len(extended_pdfs) == len(pdfs)
        if not (len(extended_pdfs) in (len(pdfs), 0)):  # either all or no extended
            raise ExtendedPDFError("Some but not all pdfs are extended. The following"
                                   "are extended \n{}\nBut gives were \n{}"
                                   "".format(extended_pdfs, pdfs))
        if all_extended:
            if frac is not None:
                raise ValueError("frac is given ({}) but all pdfs are already extended. Either"
                                 "use non-extended pdfs or give None as frac.".format(frac))
            yields = tf.stack([pdf.get_yield() for pdf in pdfs])
            frac = yields / tf.reduce_sum(yields)
        else:
            # check fraction  # TODO make more flexible, allow for Tensors and unstack
            if not len(frac) in (len(pdfs), len(pdfs) - 1):
                raise ValueError("frac has to be number of pdfs given or number of pdfs given"
                                 "minus one. Currently, frac is {} and pdfs given are {}"
                                 "".format(frac, pdfs))

            if len(frac) == len(pdfs) - 1:
                frac = list(frac) + [tf.constant(1., dtype=ztypes.float) - sum(frac)]

        super(SumPDF, self).__init__(pdfs=pdfs, frac=frac, name=name)
        if all_extended:
            self.set_yield(tf.reduce_sum(yields))

    def _unnormalized_prob(self, x):
        # TODO: deal with yields
        pdfs = self.parameters['pdfs']
        frac = self.parameters['frac']
        func = tf.accumulate_n(
            [scale * pdf.unnormalized_prob(x) for pdf, scale in zip(pdfs, tf.unstack(frac))])
        return func

    def _analytic_integrate(self, limits):
        pdfs = self.parameters['pdfs']
        frac = self.parameters['frac']
        try:
            integral = [pdf.analytic_integrate(limits) for pdf in pdfs]
        except NotImplementedError as original_error:
            raise NotImplementedError("analytic_integrate of pdf {name} is not implemented in this"
                                      " SumPDF, as at least one sub-pdf does not implement it."
                                      "Original message:\n{error}".format(name=self.name,
                                                                          error=original_error))

        integral = [integral * s for pdf, s in zip(integral, frac)]
        integral = sum(integral)  # TODO: deal with yields
        return integral


class ProductPDF(BasePDF):
    def __init__(self, pdfs, name="ProductPDF"):
        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]
        super(ProductPDF, self).__init__(pdfs=pdfs, name=name)

    def _unnormalized_prob(self, x):
        return np.prod([pdf.unnormalized_prob(x) for pdf in self.parameters['pdfs']], axis=0)


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
            vals = sum_gauss.unnormalized_prob(
                tf.convert_to_tensor(test_values, dtype=ztypes.float))
            vals = sess.run(vals)
            test_sum = sum([g.unnormalized_prob(test_values) for g in [gauss1, gauss2, gauss3]])
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

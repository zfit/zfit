"""
Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the pdfs easier.

Their implementation is often non-trivial.
"""
import tensorflow as tf
from zfit import ztf
from zfit.core.limits import no_norm_range, supports
from zfit.util import ztyping

from zfit.util.exception import ExtendedPDFError
from zfit.core.basepdf import BasePDF
from zfit.core.parameter import FitParameter
from zfit.settings import types as ztypes


class BaseFunctor(BasePDF):

    def __init__(self, pdfs, name="BaseFunctor", **kwargs):

        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]
        self.pdfs = pdfs
        super().__init__(name=name, **kwargs)


class SumPDF(BaseFunctor):

    def __init__(self, pdfs, fracs=None, name="SumPDF"):
        """

        Args:
            pdfs (zfit pdf): The basic to multiply with each other
            fracs (iterable): coefficients for the multiplication. If a scale is set, None
                will take the scale automatically. If the scale is not set, then:

                  - len(frac) = len(basic) - 1 results in the interpretation of a pdf. The last
                    coefficient equals to 1 - sum(frac)
                  - len(frac) = len(basic) results in the interpretation of multiplying each
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
            raise ExtendedPDFError("Some but not all basic are extended. The following"
                                   "are extended \n{}\nBut gives were \n{}"
                                   "".format(extended_pdfs, pdfs))
        if all_extended:
            if fracs is not None:
                raise ValueError("frac is given ({}) but all basic are already extended. Either"
                                 "use non-extended basic or give None as frac.".format(fracs))
            yields = tf.stack([pdf.get_yield() for pdf in pdfs])
            fracs = yields / tf.reduce_sum(yields)
            # fracs = [ztf.constant(1.)] * len(pdfs)
        else:
            # check fraction  # TODO make more flexible, allow for Tensors and unstack
            if not len(fracs) in (len(pdfs), len(pdfs) - 1):
                raise ValueError("frac has to be number of basic given or number of basic given"
                                 "minus one. Currently, frac is {} and basic given are {}"
                                 "".format(fracs, pdfs))

            if len(fracs) == len(pdfs) - 1:
                fracs = list(fracs) + [tf.constant(1., dtype=ztypes.float) - sum(fracs)]
            else:
                for frac, pdf in zip(fracs, pdfs):
                    pdfs.set_yield(tf.identity(frac))
                yields = tf.identity(fracs)
                fracs = fracs / tf.reduce_sum(fracs)
                all_extended = True
        super().__init__(pdfs=pdfs, fracs=fracs, name=name)
        if all_extended:
            self.set_yield(tf.reduce_sum(yields))

    def _unnormalized_prob(self, x):
        # TODO: deal with yields
        pdfs = self.pdfs
        fracs = self.parameters['fracs']
        func = tf.accumulate_n(
            [scale * pdf.unnormalized_prob(x) for pdf, scale in zip(pdfs, tf.unstack(fracs))])
        return func

    def _prob(self, x, norm_range):
        pdfs = self.pdfs
        fracs = self.parameters['fracs']
        probs = []
        for frac, pdf in zip(tf.unstack(fracs), pdfs):
            prob_pdf = pdf.prob(x, norm_range=norm_range) * frac
            if pdf.is_extended:
                prob_pdf / pdf.get_yield()
            probs.append(prob_pdf)
        prob = tf.accumulate_n(probs)
        return prob

    # def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool):
    #     return value
    #
    # @supports()
    # def _analytic_integrate(self, limits):  # TODO: deal with norm_range?
    #     pdfs = self.pdfs
    #     frac = self.parameters['frac']
    #     try:
    #         integral = [pdf.analytic_integrate(limits) for pdf in pdfs]
    #     except NotImplementedError as original_error:
    #         raise NotImplementedError("analytic_integrate of pdf {name} is not implemented in this"
    #                                   " SumPDF, as at least one sub-pdf does not implement it."
    #                                   "Original message:\n{error}".format(name=self.name,
    #                                                                       error=original_error))
    #
    #     integral = [integral * s for pdf, s in zip(integral, frac)]
    #     integral = tf.reduce_sum(integral)  # TODO: deal with yields
    #     return integral
    # def _integrate(self, limits, norm_range):



class ProductPDF(BaseFunctor):  # TODO: unfinished
    def __init__(self, pdfs, name="ProductPDF"):
        if not hasattr(pdfs, "__len__"):
            pdfs = [pdfs]
        super().__init__(pdfs=pdfs, name=name)

    def _unnormalized_prob(self, x):
        return tf.reduce_prod([pdf.unnormalized_prob(x) for pdf in self.pdfs], axis=0)


if __name__ == '__main__':

    import numpy as np
    from zfit.pdfs.basic import Gauss


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

        sum_gauss.set_norm_range = -55., 55.

        init = tf.global_variables_initializer()
        sess.run(init)


        def test_func_sum():
            test_values = np.array([3., 129., -0.2, -78.2])
            vals = sum_gauss.unnormalized_prob(
                ztf.convert_to_tensor(test_values, dtype=ztypes.float))
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

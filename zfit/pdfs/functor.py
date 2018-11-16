"""
Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the pdfs easier.

Their implementation is often non-trivial.
"""
import tensorflow as tf
from zfit import ztf
from zfit.core.limits import no_norm_range, supports
from zfit.util import ztyping
from zfit.util.container import convert_to_container

from zfit.util.exception import ExtendedPDFError
from zfit.core.basepdf import BasePDF
from zfit.core.parameter import FitParameter
from zfit.settings import types as ztypes


class BaseFunctor(BasePDF):

    def __init__(self, pdfs, name="BaseFunctor", **kwargs):
        super().__init__(name=name, **kwargs)
        pdfs = convert_to_container(pdfs)
        self.pdfs = pdfs
        self.pdfs_extended = [pdf.is_extended for pdf in pdfs]


class SumPDF(BaseFunctor):

    def __init__(self, pdfs, fracs=None, name="SumPDF"):
        """Create the sum of the `pdfs` with `fracs` as coefficients.

        Args:
            pdfs (pdf): The pdfs to add.
            fracs (iterable): coefficients for the linear combination of the pdfs. If pdfs are
                extended, this throws an error.

                  - len(frac) == len(basic) - 1 results in the interpretation of a non-extended pdf.
                    The last coefficient will equal to 1 - sum(frac)
                  - len(frac) == len(pdf) each pdf in `pdfs` will become an extended pdf with the
                    given yield.
            name (str):
        """
        # Check user input, improve TODO
        super().__init__(pdfs=pdfs, name=name)
        pdfs = self.pdfs
        if fracs is not None:
            fracs = convert_to_container(fracs)

        # check if all are extended or None is extended
        extended_pdfs = self.pdfs_extended
        all_extended = all(extended_pdfs)
        mixed_extended = not all_extended and any(extended_pdfs)  # 1 or more but not all
        if mixed_extended:  # either all or no extended
            raise ExtendedPDFError("Some but not all basic are extended. The following"
                                   "are extended \n{}\nBut gives were \n{}"
                                   "".format(extended_pdfs, pdfs))

        if all_extended:
            if fracs is not None:
                raise ValueError("frac is given ({}) but all basic are already extended. Either"
                                 "use non-extended basic or give None as frac.".format(fracs))
            else:
                yields = [pdf.get_yield() for pdf in pdfs]

        if fracs is not None and len(fracs) == len(pdfs):  # make all extended
            for pdf, frac in zip(pdfs, fracs):
                pdf.set_yield(frac)
            all_extended = True
            yields = fracs

        if all_extended:
            yield_fracs = [yield_ / tf.reduce_sum(yields) for yield_ in yields]
            self.fracs = yield_fracs
            self.set_yield(tf.reduce_sum(yields))

        else:
            # check fraction  # TODO make more flexible, allow for Tensors and unstack
            if not len(fracs) == len(pdfs) - 1:
                raise ValueError("frac has to be number of basic given or number of basic given"
                                 "minus one. Currently, frac is {} and basic given are {}"
                                 "".format(fracs, pdfs))

            fracs = list(fracs) + [tf.constant(1., dtype=ztypes.float) - sum(fracs)]
            self.fracs = fracs

        if all(self.pdfs_extended):
            self.fracs = [tf.constant(1, dtype=ztypes.float)] * len(self.pdfs)

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm_range=norm_range, log=log)

    def _unnormalized_prob(self, x):
        # TODO: deal with yields
        pdfs = self.pdfs
        fracs = self.fracs
        func = tf.accumulate_n(
            [scale * pdf.unnormalized_prob(x) for pdf, scale in zip(pdfs, fracs)])
        return func

    def _prob(self, x, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        prob = tf.accumulate_n([pdf.prob(x, norm_range=norm_range) * scale for pdf, scale in zip(pdfs, fracs)])
        return prob

    @supports()
    def _analytic_integrate(self, limits):  # TODO: deal with norm_range?
        pdfs = self.pdfs
        frac = self.fracs
        try:
            integral = [pdf.analytic_integrate(limits) for pdf in pdfs]
        except NotImplementedError as original_error:
            raise NotImplementedError("analytic_integrate of pdf {name} is not implemented in this"
                                      " SumPDF, as at least one sub-pdf does not implement it."
                                      "Original message:\n{error}".format(name=self.name,
                                                                          error=original_error))

        integral = tf.stack([integral * s for pdf, s in zip(integral, frac)])
        integral = tf.reduce_sum(integral)
        return integral

    @supports()
    def _partial_analytic_integrate(self, x, limits):
        pdfs = self.pdfs
        frac = self.fracs
        try:
            partial_integral = [pdf.analytic_integrate(limits) for pdf in pdfs]
        except NotImplementedError as original_error:
            raise NotImplementedError("partial_analytic_integrate of pdf {name} is not implemented in this"
                                      " SumPDF, as at least one sub-pdf does not implement it."
                                      "Original message:\n{error}".format(name=self.name,
                                                                          error=original_error))
        partial_integral = tf.stack([partial_integral * s for pdf, s in zip(partial_integral, frac)])
        partial_integral = tf.reduce_sum(partial_integral, axis=0)
        return partial_integral


class ProductPDF(BaseFunctor):  # TODO: unfinished
    def __init__(self, pdfs, dims=None, name="ProductPDF"):
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

"""
Functors are functions that take typically one or more other PDF. Prominent examples are a sum, convolution etc.

A FunctorBase class is provided to make handling the models easier.

Their implementation is often non-trivial.
"""
import itertools
from typing import Union, List, Optional

import tensorflow as tf
from zfit import ztf
from zfit.core.dimension import get_same_dims, unstack_x_dims
from zfit.core.interfaces import ZfitPDF, ZfitModel
from zfit.core.limits import no_norm_range, supports
from zfit.models.basefunctor import FunctorMixin
from zfit.util import ztyping
from zfit.util.container import convert_to_container

from zfit.util.exception import ExtendedPDFError, AlreadyExtendedPDFError, DimsNotUnambiguousError
from zfit.core.basepdf import BasePDF
from zfit.core.parameter import Parameter
from zfit.settings import types as ztypes


class BaseFunctor(FunctorMixin, BasePDF):

    def __init__(self, pdfs, name="BaseFunctor", **kwargs):
        self.pdfs = convert_to_container(pdfs)
        super().__init__(models=self.pdfs, name=name, **kwargs)

    @property
    def pdfs_extended(self):
        return [pdf.is_extended for pdf in self.pdfs]

    @property
    def _models(self) -> List[ZfitModel]:
        return self.pdfs


class SumPDF(BaseFunctor):

    def __init__(self, pdfs: List[ZfitPDF], fracs: Optional[List[float]] = None, dims: ztyping.DimsType = None,
                 name: str = "SumPDF") -> "SumPDF":
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
        super().__init__(dims=dims, pdfs=pdfs, name=name)
        pdfs = self.pdfs
        if len(pdfs) < 2:
            raise ValueError("Cannot build a sum of a single pdf")
        self.dims = dims
        if fracs is not None:
            fracs = convert_to_container(fracs)
            fracs = [ztf.convert_to_tensor(frac) for frac in fracs]

        # check if all extended
        extended_pdfs = self.pdfs_extended
        implicit = None
        extended = None
        if all(extended_pdfs):
            implicit = True
            extended = True

        # all extended except one -> fraction
        elif sum(extended_pdfs) == len(extended_pdfs) - 1:
            implicit = True
            extended = False

        # no pdf is extended -> using `fracs`
        elif not any(extended_pdfs) and fracs is not None:
            # make extended
            if len(fracs) == len(pdfs):
                implicit = False
                extended = True
            elif len(fracs) == len(pdfs) - 1:
                implicit = False
                extended = False

        # catch if args don't fit known case
        value_error = implicit is None or extended is None
        if (implicit and fracs is not None) or value_error:
            raise TypeError("Wrong arguments. Either"
                            "\n a) `pdfs` are not extended and `fracs` is given with length pdfs "
                            "(-> pdfs get extended) or pdfs - 1 (fractions)"
                            "\n b) all or all except 1 `pdfs` are extended and fracs is None.")

        # create fracs if one is not extended
        if not extended and implicit:
            fracs = []
            not_extended_position = None
            for i, pdf in enumerate(pdfs):
                if pdf.is_extended:
                    fracs.append(pdf.get_yield())
                    pdf.set_yield(None)  # make non-extended
                else:
                    fracs.append(tf.constant(0., dtype=ztypes.float))
                    not_extended_position = i
            remaining_frac = tf.constant(1., dtype=ztypes.float) - tf.add_n(fracs)
            assert_op = tf.Assert(tf.greater_equal(remaining_frac, tf.constant(0., dtype=ztypes.float)),
                                  data=[remaining_frac])  # check fractions
            with tf.control_dependencies([assert_op]):
                fracs[not_extended_position] = tf.identity(remaining_frac)
            implicit = False  # now it's explicit

        elif not extended and not implicit:
            remaining_frac = tf.constant(1., dtype=ztypes.float) - tf.add_n(fracs)
            assert_op = tf.Assert(tf.greater_equal(remaining_frac, tf.constant(0., dtype=ztypes.float)),
                                  data=[remaining_frac])  # check fractions
            with tf.control_dependencies([assert_op]):
                fracs.append(tf.identity(remaining_frac))

        # make extended
        elif extended and not implicit:
            yields = fracs
            for pdf, yield_ in zip(pdfs, yields):
                pdf.set_yield(yield_)
            implicit = True

        elif extended and implicit:
            yields = [pdf.get_yield() for pdf in pdfs]

        if extended:
            # yield_fracs = [yield_ / tf.reduce_sum(yields) for yield_ in yields]
            # self.fracs = yield_fracs
            self.set_yield(tf.reduce_sum(yields))
            self.fracs = [tf.constant(1, dtype=ztypes.float)] * len(self.pdfs)
        else:
            self.fracs = fracs

    @property
    def _functor_allow_none_dims(self) -> bool:
        return True

    @property
    def _n_dims(self):
        return 1  # TODO(mayou36): properly implement dimensions

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool):
        if all(self.pdfs_extended):
            return value
        else:
            return super()._apply_yield(value=value, norm_range=norm_range, log=log)

    def _unnormalized_pdf(self, x, norm_range=False):
        raise NotImplementedError
        # TODO: deal with yields
        # pdfs = self.pdfs
        # fracs = self.fracs
        # func = tf.accumulate_n(
        #     [scale * pdf.unnormalized_pdf(x) for pdf, scale in zip(pdfs, fracs)])
        # return func

    def _pdf(self, x, norm_range):
        pdfs = self.pdfs
        fracs = self.fracs
        prob = tf.add_n([pdf.pdf(x, norm_range=norm_range) * scale for pdf, scale in zip(pdfs, fracs)])
        # prob = tf.accumulate_n([pdf.pdf(x, norm_range=norm_range) * scale for pdf, scale in zip(pdfs, fracs)])
        return prob

    def _set_yield(self, value: Union[Parameter, None]):
        # TODO: what happens now with the daugthers?
        if all(
            self.pdfs_extended) and self.is_extended and value is not None:  # to be able to set the yield in the
            # beginning
            raise AlreadyExtendedPDFError("Cannot set the yield of a PDF with extended daughters.")
        elif all(self.pdfs_extended) and self.is_extended and value is None:  # not extended anymore
            reciprocal_yield = tf.reciprocal(self.get_yield())
            self.fracs = [reciprocal_yield] * len(self.fracs)
        else:
            super()._set_yield(value=value)

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
    def __init__(self, pdfs, dims, name="ProductPDF"):
        super().__init__(dims=dims, pdfs=pdfs, name=name)

    @property
    def _functor_allow_none_dims(self) -> bool:
        return False

    def _unnormalized_pdf(self, x, norm_range=False):
        x_unstacked = unstack_x_dims(x=x, dims=self._model_dims_index)
        return tf.reduce_prod([pdf.unnormalized_pdf(x) for x, pdf in zip(x_unstacked, self.pdfs)], axis=0)

    def _pdf(self, x, norm_range):
        if all(not dep for dep in self._model_same_dims):
            x_unstacked = unstack_x_dims(x=x, dims=self._model_dims_index)
            probs = [pdf.pdf(x=x, norm_range=norm_range.subspace(dims=pdf.dims)) for x, pdf in
                     zip(x_unstacked, self.pdfs)]
            return tf.reduce_prod(probs, axis=0)
        else:
            raise NotImplementedError


if __name__ == '__main__':

    import numpy as np
    from zfit.models.basic import Gauss


    def true_gaussian_sum(x):
        sum_gauss = np.exp(- (x - 1.) ** 2 / (2 * 11. ** 2))
        sum_gauss += np.exp(- (x - 2.) ** 2 / (2 * 22. ** 2))
        sum_gauss += np.exp(- (x - 3.) ** 2 / (2 * 33. ** 2))
        return sum_gauss


    with tf.Session() as sess:
        mu1 = Parameter("mu1", 1.)
        mu2 = Parameter("mu2", 2.)
        mu3 = Parameter("mu3", 3.)
        sigma1 = Parameter("sigma1", 11.)
        sigma2 = Parameter("sigma2", 22.)
        sigma3 = Parameter("sigma3", 33.)

        gauss1 = Gauss(mu=mu1, sigma=sigma1, name="gauss1")
        gauss2 = Gauss(mu=mu2, sigma=sigma2, name="gauss2")
        gauss3 = Gauss(mu=mu3, sigma=sigma3, name="gauss3")

        sum_gauss = SumPDF(pdfs=[gauss1, gauss2, gauss3])

        sum_gauss.set_norm_range = -55., 55.

        init = tf.global_variables_initializer()
        sess.run(init)


        def test_func_sum():
            test_values = np.array([3., 129., -0.2, -78.2])
            vals = sum_gauss.unnormalized_pdf(
                ztf.convert_to_tensor(test_values, dtype=ztypes.float))
            vals = sess.run(vals)
            test_sum = sum([g.unnormalized_pdf(test_values) for g in [gauss1, gauss2, gauss3]])
            print(sess.run(test_sum))
            np.testing.assert_almost_equal(vals, true_gaussian_sum(test_values))


        def test_normalization():
            low, high = sum_gauss.norm_range
            samples = tf.cast(np.random.uniform(low=low, high=high, size=100000),
                              dtype=tf.float64)
            samples.limits = low, high
            probs = sum_gauss.pdf(samples)
            result = sess.run(probs)
            result = np.average(result) * (high - low)
            print(result)
            assert 0.95 < result < 1.05


        test_func_sum()
        test_normalization()

import numpy as np

import tensorflow as tf

from zfit import Parameter, ztf
from zfit.core.parameter import ComposedParameter
from zfit.models.functions import SimpleFunction
from zfit.models.functor import SumPDF
from zfit.models.special import SimplePDF

rnd_test_values = np.array([1., 0.01, -14.2, 0., 1.5, 152, -0.1, 12])


def test_composed_param():
    # tf.reset_default_graph()
    param1 = Parameter('param1s', 1.)
    param2 = Parameter('param2s', 2.)
    param3 = Parameter('param3s', 3., floating=False)
    param4 = Parameter('param4s', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    param_a = ComposedParameter('param_as', tensor=a)
    assert isinstance(param_a.get_dependents(only_floating=True), set)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}


def test_param_func():
    param1 = Parameter('param11s', 1.)
    param2 = Parameter('param21s', 2.)
    param3 = Parameter('param31s', 3., floating=False)
    param4 = Parameter('param41s', 4.)
    # a = ztf.log(3. * param1) * tf.square(param2) - param3
    a = 3. * param1
    func = SimpleFunction(func=lambda x: a * x)

    new_func = param4 * func

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result1 = sess.run(new_func.value(x=rnd_test_values))
        result2 = sess.run(func.value(x=rnd_test_values) * param4)
        assert all(result1 == result2)


def test_implicit_extended():
    # tf.reset_default_graph()
    param1 = Parameter('param12s', 12.)
    yield1 = Parameter('yield12s', 21.)
    param2 = Parameter('param22s', 13., floating=False)
    yield2 = Parameter('yield22s', 31., floating=False)
    pdf1 = SimplePDF(func=lambda x: x * param1)
    pdf2 = SimplePDF(func=lambda x: x * param2)
    extended_pdf = yield1 * pdf1 + yield2 * pdf2

    true_extended_pdf = SumPDF(pdfs=[pdf1, pdf2])
    assert isinstance(extended_pdf, SumPDF)
    assert extended_pdf.is_extended
    assert true_extended_pdf == extended_pdf


def test_implicit_sumpdf():
    # tf.reset_default_graph()
    norm_range = (-5.7, 13.6)
    param1 = Parameter('param23s', 1.1)
    frac1 = 0.11
    frac1_param = Parameter('frac13s', frac1)
    frac2 = 0.66
    frac2_param = Parameter('frac23s', frac2)
    frac3 = 1 - frac1 - frac2  # -frac1_param -frac2_param

    param2 = Parameter('param23s', 1.5, floating=False)
    param3 = Parameter('param33s', 0.4, floating=False)
    pdf1 = SimplePDF(func=lambda x: x * param1**2)
    pdf2 = SimplePDF(func=lambda x: x * param2)
    pdf3 = SimplePDF(func=lambda x: x * 2 + param3)

    # sugar 1
    sum_pdf = frac1_param * pdf1 + pdf2 + frac2_param * pdf3


    true_values = pdf1.pdf(rnd_test_values, norm_range=norm_range)
    true_values *= frac1_param
    true_values += pdf2.pdf(rnd_test_values, norm_range=norm_range) * tf.constant(frac3, dtype=tf.float64)
    true_values += pdf3.pdf(rnd_test_values, norm_range=norm_range) * frac2_param

    assert isinstance(sum_pdf, SumPDF)
    assert not sum_pdf.is_extended
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert sess.run(sum(sum_pdf.fracs)) == 1.
        true_values = sess.run(true_values)
        test_values = sess.run(sum_pdf.pdf(rnd_test_values, norm_range=norm_range))
        np.testing.assert_allclose(true_values, test_values, rtol=1e-2)  # it's MC normalized

        # sugar 2
        sum_pdf2_part1 = frac1 * pdf1 + frac2 * pdf3
        sum_pdf2 = sum_pdf2_part1 + pdf2

        sess.run(tf.global_variables_initializer())
        test_values2 = sess.run(sum_pdf2.pdf(rnd_test_values, norm_range=norm_range))
        np.testing.assert_allclose(true_values, test_values2, rtol=1e-2)  # it's MC normalized

import numpy as np
import pytest

import tensorflow as tf

import zfit
from zfit import Parameter, ztf
from zfit.models.functions import SimpleFunction
from zfit.models.functor import SumPDF
from zfit.models.special import SimplePDF
from zfit.util.exception import LogicalUndefinedOperationError, AlreadyExtendedPDFError

rnd_test_values = np.array([1., 0.01, -14.2, 0., 1.5, 152, -0.1, 12])


def test_not_allowed():
    param1 = Parameter('param11sda', 1.)
    param2 = Parameter('param21dsa', 2.)
    param3 = Parameter('param31sda', 3., floating=False)
    param4 = Parameter('param41sda', 4.)

    def func1_pure(x):
        return param1 * x

    def func2_pure(x):
        return param2 * x + param3

    func1 = SimpleFunction(func=func1_pure, p1=param1)
    func2 = SimpleFunction(func=func2_pure, p2=param2, p3=param3)

    pdf1 = SimplePDF(func=lambda x: x * param1)
    pdf2 = SimplePDF(func=lambda x: x * param2)

    with pytest.raises(TypeError):
        pdf1 + pdf2
    with pytest.raises(TypeError):
        pdf1 + pdf2
    with pytest.raises(NotImplementedError):
        param1 + func1
    with pytest.raises(NotImplementedError):
        func1 + param1
    with pytest.raises(TypeError):
        func1 * pdf2
    with pytest.raises(TypeError):
        pdf1 * func1


def test_param_func():
    param1 = Parameter('param11s', 1.)
    param2 = Parameter('param21s', 2.)
    param3 = Parameter('param31s', 3., floating=False)
    param4 = Parameter('param41s', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    # a = 3. * param1
    func = SimpleFunction(func=lambda x: a * x)

    new_func = param4 * func

    new_func_equivalent = func * param4

    zfit.sess.run(tf.global_variables_initializer())
    result1 = zfit.sess.run(new_func.value(x=rnd_test_values))
    result1_equivalent = zfit.sess.run(new_func_equivalent.value(x=rnd_test_values))
    result2 = zfit.sess.run(func.value(x=rnd_test_values) * param4)
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1_equivalent, result2)


def test_func_func():
    param1 = Parameter('param11sd', 1.)
    param2 = Parameter('param21ds', 2.)
    param3 = Parameter('param31sd', 3., floating=False)
    param4 = Parameter('param41sd', 4.)

    def func1_pure(x):
        return param1 * x

    def func2_pure(x):
        return param2 * x + param3

    func1 = SimpleFunction(func=func1_pure, p1=param1)
    func2 = SimpleFunction(func=func2_pure, p2=param2, p3=param3)

    added_func = func1 + func2
    prod_func = func1 * func2

    added_values = added_func.value(rnd_test_values)
    true_added_values = func1_pure(rnd_test_values) + func2_pure(rnd_test_values)
    prod_values = prod_func.value(rnd_test_values)
    true_prod_values = func1_pure(rnd_test_values) * func2_pure(rnd_test_values)

    zfit.sess.run(tf.global_variables_initializer())
    # added_values = zfit.sess.run(added_values)
    true_added_values = zfit.sess.run(true_added_values)
    prod_values = zfit.sess.run(prod_values)
    true_prod_values = zfit.sess.run(true_prod_values)
    # np.testing.assert_allclose(true_added_values, added_values)
    np.testing.assert_allclose(true_prod_values, prod_values)


def test_param_pdf():
    param1 = Parameter('param12sa', 12.)
    param2 = Parameter('param22sa', 22.)
    yield1 = Parameter('yield12sa', 21.)
    yield2 = Parameter('yield22sa', 22.)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    pdf1 = SimplePDF(func=lambda x: x * param1)
    pdf2 = SimplePDF(func=lambda x: x * param2)
    assert not pdf1.is_extended
    extended_pdf = yield1 * pdf1
    assert extended_pdf.is_extended
    with pytest.raises(TypeError):
        _ = pdf2 * yield2
    with pytest.raises(AlreadyExtendedPDFError):
        _ = yield2 * extended_pdf


def test_implicit_extended():
    # tf.reset_default_graph()
    param1 = Parameter('param12s', 12.)
    yield1 = Parameter('yield12s', 21.)
    param2 = Parameter('param22s', 13., floating=False)
    yield2 = Parameter('yield22s', 31., floating=False)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
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
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    pdf1 = SimplePDF(func=lambda x: x * param1 ** 2)
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

    zfit.sess.run(tf.global_variables_initializer())
    assert zfit.sess.run(sum(sum_pdf.fracs)) == 1.
    true_values = zfit.sess.run(true_values)
    test_values = zfit.sess.run(sum_pdf.pdf(rnd_test_values, norm_range=norm_range))
    np.testing.assert_allclose(true_values, test_values, rtol=1e-2)  # it's MC normalized

    # sugar 2
    sum_pdf2_part1 = frac1 * pdf1 + frac2 * pdf3
    sum_pdf2 = sum_pdf2_part1 + pdf2

    zfit.sess.run(tf.global_variables_initializer())
    test_values2 = zfit.sess.run(sum_pdf2.pdf(rnd_test_values, norm_range=norm_range))
    np.testing.assert_allclose(true_values, test_values2, rtol=1e-2)  # it's MC normalized

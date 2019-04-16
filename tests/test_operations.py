import numpy as np
import pytest

import tensorflow as tf

import zfit
from zfit import Parameter, ztf
from zfit.models.functions import SimpleFunc
from zfit.models.functor import SumPDF
from zfit.models.special import SimplePDF
from zfit.util.exception import LogicalUndefinedOperationError, AlreadyExtendedPDFError, ModelIncompatibleError
from zfit.core.testing import setup_function, teardown_function, tester

rnd_test_values = np.array([1., 0.01, -14.2, 0., 1.5, 152, -0.1, 12])

obs1 = 'obs1'


def test_not_allowed():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)

    def func1_pure(self, x):
        return param1 * x

    def func2_pure(self, x):
        return param2 * x + param3

    func1 = SimpleFunc(func=func1_pure, obs=obs1, p1=param1)
    func2 = SimpleFunc(func=func2_pure, obs=obs1, p2=param2, p3=param3)

    pdf1 = SimplePDF(func=lambda self, x: x * param1, obs=obs1)
    pdf2 = SimplePDF(func=lambda self, x: x * param2, obs=obs1)

    with pytest.raises(ModelIncompatibleError):
        pdf1 + pdf2
    with pytest.raises(NotImplementedError):
        param1 + func1
    with pytest.raises(NotImplementedError):
        func1 + param1
    with pytest.raises(ModelIncompatibleError):
        func1 * pdf2
    with pytest.raises(ModelIncompatibleError):
        pdf1 * func1


def test_param_func():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    func = SimpleFunc(func=lambda self, x: a * x, obs=obs1)

    new_func = param4 * func

    new_func_equivalent = func * param4

    result1 = zfit.run(new_func.func(x=rnd_test_values))
    result1_equivalent = zfit.run(new_func_equivalent.func(x=rnd_test_values))
    result2 = zfit.run(func.func(x=rnd_test_values) * param4)
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1_equivalent, result2)


def test_func_func():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)

    def func1_pure(self, x):
        x = ztf.unstack_x(x)
        return param1 * x

    def func2_pure(self, x):
        x = ztf.unstack_x(x)
        return param2 * x + param3

    func1 = SimpleFunc(func=func1_pure, obs=obs1, p1=param1)
    func2 = SimpleFunc(func=func2_pure, obs=obs1, p2=param2, p3=param3)

    added_func = func1 + func2
    prod_func = func1 * func2

    added_values = added_func.func(rnd_test_values)
    true_added_values = func1_pure(None, rnd_test_values) + func2_pure(None, rnd_test_values)
    prod_values = prod_func.func(rnd_test_values)
    true_prod_values = func1_pure(None, rnd_test_values) * func2_pure(None, rnd_test_values)

    added_values = zfit.run(added_values)
    true_added_values = zfit.run(true_added_values)
    prod_values = zfit.run(prod_values)
    true_prod_values = zfit.run(true_prod_values)
    np.testing.assert_allclose(true_added_values, added_values)
    np.testing.assert_allclose(true_prod_values, prod_values)


def test_param_pdf():
    # return  # TODO(Mayou36): deps: impl_copy,
    param1 = Parameter('param1', 12.)
    param2 = Parameter('param2', 22.)
    yield1 = Parameter('yield1', 21.)
    yield2 = Parameter('yield2', 22.)
    pdf1 = SimplePDF(func=lambda self, x: x * param1, obs=obs1)
    pdf2 = SimplePDF(func=lambda self, x: x * param2, obs=obs1)
    assert not pdf1.is_extended
    extended_pdf = yield1 * pdf1
    assert extended_pdf.is_extended
    with pytest.raises(ModelIncompatibleError):
        _ = pdf2 * yield2
    with pytest.raises(AlreadyExtendedPDFError):
        _ = yield2 * extended_pdf


def test_implicit_extended():
    # return  # TODO(Mayou36): deps: impl_copy,

    param1 = Parameter('param1', 12.)
    yield1 = Parameter('yield1', 21.)
    param2 = Parameter('param2', 13., floating=False)
    yield2 = Parameter('yield2', 31., floating=False)
    pdf1 = SimplePDF(func=lambda self, x: x * param1, obs=obs1)
    pdf2 = SimplePDF(func=lambda self, x: x * param2, obs=obs1)
    extended_pdf = yield1 * pdf1 + yield2 * pdf2
    with pytest.raises(ModelIncompatibleError):
        true_extended_pdf = SumPDF(pdfs=[pdf1, pdf2], obs=obs1)
    assert isinstance(extended_pdf, SumPDF)
    assert extended_pdf.is_extended


def test_implicit_sumpdf():
    # return  # TODO(Mayou36): deps: impl_copy, (mostly for Simple{PDF,Func})

    norm_range = (-5.7, 13.6)
    param1 = Parameter('param13s', 1.1)
    frac1 = 0.11
    frac1_param = Parameter('frac13s', frac1)
    frac2 = 0.56
    frac2_param = Parameter('frac23s', frac2)
    frac3 = 1 - frac1 - frac2  # -frac1_param -frac2_param

    param2 = Parameter('param23s', 1.5, floating=False)
    param3 = Parameter('param33s', 0.4, floating=False)
    pdf1 = SimplePDF(func=lambda self, x: x * param1 ** 2, obs=obs1)
    pdf2 = SimplePDF(func=lambda self, x: x * param2, obs=obs1)
    pdf3 = SimplePDF(func=lambda self, x: x * 2 + param3, obs=obs1)

    # sugar 1
    # sum_pdf = frac1_param * pdf1 + frac2_param * pdf2 + pdf3  # TODO(Mayou36): deps, correct copy
    sum_pdf = zfit.pdf.SumPDF(pdfs=[pdf1, pdf2, pdf3], fracs=[frac1_param, frac2_param])

    true_values = pdf1.pdf(rnd_test_values, norm_range=norm_range)
    true_values *= frac1_param
    true_values += pdf2.pdf(rnd_test_values, norm_range=norm_range) * frac2_param
    true_values += pdf3.pdf(rnd_test_values, norm_range=norm_range) * ztf.constant(frac3)

    assert isinstance(sum_pdf, SumPDF)
    assert not sum_pdf.is_extended

    assert zfit.run(sum(sum_pdf._maybe_extended_fracs)) == 1.
    true_values = zfit.run(true_values)
    test_values = zfit.run(sum_pdf.pdf(rnd_test_values, norm_range=norm_range))
    np.testing.assert_allclose(true_values, test_values, rtol=5e-2)  # it's MC normalized

    # sugar 2
    sum_pdf2_part1 = frac1 * pdf1 + frac2 * pdf3
    # sum_pdf2 = sum_pdf2_part1 + pdf2  # TODO(Mayou36): deps copy

    # test_values2 = zfit.run(sum_pdf2.pdf(rnd_test_values, norm_range=norm_range))  # TODO(Mayou36): deps copy
    # np.testing.assert_allclose(true_values, test_values2, rtol=1e-2)  # it's MC normalized  # TODO(Mayou36): deps copy

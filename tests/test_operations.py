import numpy as np

import tensorflow as tf

from zfit import Parameter, ztf
from zfit.core.parameter import ComposedParameter
from zfit.pdfs.functions import SimpleFunction
from zfit.pdfs.functor import SumPDF
from zfit.pdfs.special import SimplePDF

rnd_test_values = np.array([1., 0.01,-14.2, 0., 1.5, 152, -0.1, 12])

def test_composed_param():
    # tf.reset_default_graph()
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    param_a = ComposedParameter('param_a', tensor=a)
    assert isinstance(param_a.get_dependents(only_floating=True), set)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}

def test_param_func():
    param1 = Parameter('param11', 1.)
    param2 = Parameter('param21', 2.)
    param3 = Parameter('param31', 3., floating=False)
    param4 = Parameter('param41', 4.)
    # a = ztf.log(3. * param1) * tf.square(param2) - param3
    a = 3. * param1
    func = SimpleFunction(func=lambda x: a * x)

    new_func = param4 * func

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        result1 = sess.run(new_func.value(x=rnd_test_values))
        result2 = sess.run(func.value(x=rnd_test_values) * param4.read_value())
        assert all(result1 == result2)

def test_implicit_extended():
    # tf.reset_default_graph()
    param1 = Parameter('param12', 12.)
    yield1 = Parameter('yield12', 21.)
    param2 = Parameter('param22', 13., floating=False)
    yield2 = Parameter('yield22', 31., floating=False)
    pdf1 = SimplePDF(func=lambda x: x*param1)
    pdf2 = SimplePDF(func=lambda x: x*param2)
    extended_pdf = yield1 * pdf1 + yield2 * pdf2
    assert isinstance(extended_pdf, SumPDF)
    assert extended_pdf.is_extended


def test_implicit_sumpdf():
    # tf.reset_default_graph()
    param1 = Parameter('param23', 11.)
    frac1 = Parameter('frac13', 0.17)
    frac2 = Parameter('frac23', 0.2)

    param2 = Parameter('param23', 12., floating=False)
    param3 = Parameter('param33', 13., floating=False)
    pdf1 = SimplePDF(func=lambda x: x * param1)
    pdf2 = SimplePDF(func=lambda x: x * param2)
    pdf3 = SimplePDF(func=lambda x: x * param3)
    sum_pdf = frac1 * pdf1 +  pdf2 + frac2 * pdf3
    assert isinstance(sum_pdf, SumPDF)
    assert not sum_pdf.is_extended
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert sess.run(sum(sum_pdf.fracs)) == 1.




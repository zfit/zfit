#  Copyright (c) 2022 zfit
import numpy as np
import pytest
import tensorflow as tf

from zfit import Parameter, z
from zfit.models.functions import SimpleFuncV1
from zfit.models.special import SimplePDF
from zfit.util.exception import BreakingAPIChangeError, ModelIncompatibleError

rnd_test_values = np.array([1.0, 0.01, -14.2, 0.0, 1.5, 152, -0.1, 12])

obs1 = "obs1"


def test_not_allowed():
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = Parameter("param3", 3.0, floating=False)

    def func1_pure(self, x):
        return param1 * x

    def func2_pure(self, x):
        return param2 * x + param3

    func1 = SimpleFuncV1(func=func1_pure, obs=obs1, p1=param1)

    pdf1 = SimplePDF(func=lambda self, x: x * param1, obs=obs1)
    pdf2 = SimplePDF(func=lambda self, x: x * param2, obs=obs1)

    with pytest.raises(BreakingAPIChangeError):
        pdf1 + pdf2

    ext_pdf1 = pdf1.create_extended(param1)
    with pytest.raises(BreakingAPIChangeError):
        ext_pdf1 + pdf2
    with pytest.raises(BreakingAPIChangeError):
        param2 * pdf1
    with pytest.raises(NotImplementedError):
        param1 + func1
    with pytest.raises(NotImplementedError):
        func1 + param1
    with pytest.raises(ModelIncompatibleError):
        func1 * pdf2
    with pytest.raises(ModelIncompatibleError):
        pdf1 * func1


def test_param_func():
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = Parameter("param3", 3.0, floating=False)
    param4 = Parameter("param4", 4.0)
    a = z.math.log(3.0 * param1) * tf.square(param2) - param3
    func = SimpleFuncV1(func=lambda self, x: a * x.value()[:, 0], obs=obs1)

    new_func = param4 * func

    new_func_equivalent = func * param4

    result1 = new_func.func(x=rnd_test_values).numpy()
    result1_equivalent = new_func_equivalent.func(x=rnd_test_values).numpy()
    result2 = func.func(x=rnd_test_values) * param4
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1_equivalent, result2)


def test_func_func():
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = Parameter("param3", 3.0, floating=False)

    def func1_pure(self, x):
        x = z.unstack_x(x)
        return param1 * x

    def func2_pure(self, x):
        x = z.unstack_x(x)
        return param2 * x + param3

    func1 = SimpleFuncV1(func=func1_pure, obs=obs1, p1=param1)
    func2 = SimpleFuncV1(func=func2_pure, obs=obs1, p2=param2, p3=param3)

    added_func = func1 + func2
    prod_func = func1 * func2

    added_values = added_func.func(rnd_test_values)
    true_added_values = func1_pure(None, rnd_test_values) + func2_pure(
        None, rnd_test_values
    )
    prod_values = prod_func.func(rnd_test_values)
    true_prod_values = func1_pure(None, rnd_test_values) * func2_pure(
        None, rnd_test_values
    )

    added_values = added_values.numpy()
    true_added_values = true_added_values.numpy()
    prod_values = prod_values.numpy()
    true_prod_values = true_prod_values.numpy()
    np.testing.assert_allclose(true_added_values, added_values)
    np.testing.assert_allclose(true_prod_values, prod_values)

#  Copyright (c) 2020 zfit
import numpy as np

import zfit
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.z.math import numerical_gradient, autodiff_gradient, numerical_hessian, autodiff_hessian


def test_numerical_gradient():
    param1 = zfit.Parameter('param1', 4.)
    param2 = zfit.Parameter('param2', 5.)
    param3 = zfit.Parameter('param3', 2.)

    def func1():
        return param1 * param2 ** 2 + param3 ** param1

    num_gradients = numerical_gradient(func1, params=[param1, param2, param3])
    tf_gradients = autodiff_gradient(func1, params=[param1, param2, param3])
    np.testing.assert_allclose(num_gradients, tf_gradients)


def test_numerical_hessian():
    param1 = zfit.Parameter('param1', 4.)
    param2 = zfit.Parameter('param2', 5.)
    param3 = zfit.Parameter('param3', 2.)

    def func1():
        return param1 * param2 ** 2 + param3 ** param1

    num_hessian = numerical_hessian(func1, params=[param1, param2, param3])
    tf_hessian = autodiff_hessian(func1, params=[param1, param2, param3])
    np.testing.assert_allclose(num_hessian, tf_hessian, rtol=1e-5, atol=1e-10)

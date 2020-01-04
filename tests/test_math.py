#  Copyright (c) 2020 zfit


import pytest
import tensorflow as tf
import numpy as np

import zfit
from zfit.core.math import numerical_gradient, automatic_gradient, numerical_hessian, automatic_hessian

from zfit.core.testing import setup_function, teardown_function, tester


def test_numerical_gradient():
    param1 = zfit.Parameter('param1', 4.)

    def func1():
        return param1 ** 2

    num_gradients = numerical_gradient(func1, params=param1)
    tf_gradients = automatic_gradient(func1, params=param1)
    np.testing.assert_allclose(num_gradients, tf_gradients)


def test_numerical_hessian():
    param1 = zfit.Parameter('param1', 4.)
    param2 = zfit.Parameter('param2', 5.)
    param3 = zfit.Parameter('param3', 2.)

    def func1():
        return param1 * param2 ** 2 + param3 ** param1

    num_hessian = numerical_hessian(func1, params=param1)
    tf_hessian = automatic_hessian(func1, params=param1)
    np.testing.assert_allclose(num_hessian, tf_hessian)

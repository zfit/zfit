#  Copyright (c) 2019 zfit


import pytest
import tensorflow as tf
import numpy as np

import zfit
from zfit.core.math import numerical_gradient, automatic_gradient

from zfit.core.testing import setup_function, teardown_function, tester


def test_numerical_gradient():
    import zfit
    param1 = zfit.Parameter('param1', 4.)

    def func1():
        return param1 ** 2

    num_gradients = numerical_gradient(func1, params=param1)
    tf_gradients = automatic_gradient(func1, params=param1)
    np.testing.assert_allclose(num_gradients, tf_gradients)

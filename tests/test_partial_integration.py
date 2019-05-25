#  Copyright (c) 2019 zfit
import pytest
import scipy
import scipy.special
import numpy as np
import tensorflow as tf

from zfit.core.testing import setup_function, teardown_function, tester


import zfit


def func_cosxy2_np(x, y):
    return np.cos(x * y ** 2)


def func_cosxy2_tf(x, y):
    return tf.cos(x * y ** 2)


def integral_y(x, lowery, uppery):
    def indef_limit(limit):
        integral = (np.sqrt(2) * np.sqrt(np.pi) * scipy.special.fresnel(
            np.sqrt(2) * limit * np.sqrt(x) / np.sqrt(np.pi))[1] * scipy.special.gamma(1 / 4) / (
                        8 * np.sqrt(x) * scipy.special.gamma(5 / 4)))
        return integral

    return indef_limit(uppery) - indef_limit(lowery)


def integral_x(y, lowerx, upperx):
    def indef_limit(limit):
        return np.sin(limit * y ** 2) / y ** 2

    return indef_limit(upperx) - indef_limit(lowerx)


def test_partial_integral():
    class CosXY2(zfit.pdf.ZPDF):
        _PARAMS = []

        def _unnormalized_pdf(self, x):
            x, y = x.unstack_x()
            return func_cosxy2_tf(x=x, y=y)

    xspace = zfit.Space("x", limits=(1, 4))
    yspace = zfit.Space("y", limits=(1, 4))
    obs = xspace * yspace

    cosxy2 = CosXY2(obs=obs)

    data_np = np.linspace((1, 1), (4, 4), 4000)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)
    probs = cosxy2.pdf(data)
    probs_np = zfit.run(probs)
    x = data_np[:, 0]
    y = data_np[:, 1]
    probs_func = func_cosxy2_np(x=x, y=y)
    ratios = probs_np / probs_func
    assert pytest.approx(0., rel=0.0001) == np.std(ratios)  # ratio should be constant
    ratio = np.average(ratios)

    integral_x_tf = cosxy2.partial_integrate(x=data, limits=xspace)
    integral_y_tf = cosxy2.partial_integrate(x=data, limits=yspace)
    integral_x_np = zfit.run(integral_x_tf)
    integral_y_np = zfit.run(integral_y_tf)

    lowerx, upperx = xspace.limit1d
    lowery, uppery = yspace.limit1d
    integral_x_true = integral_x(y=y, lowerx=lowerx, upperx=upperx) * ratio
    integral_y_true = integral_y(x=x, lowery=lowery, uppery=uppery) * ratio

    np.testing.assert_allclose(integral_x_true, integral_x_np, atol=3.5e-3)
    np.testing.assert_allclose(integral_y_true, integral_y_np, atol=3.5e-3)

#  Copyright (c) 2020 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
import zfit.models.dist_tfp
from zfit import ztypes


def test_simple_kde():
    expected_integral = 5 / 6
    h = zfit.Parameter("h", 0.9)

    size = 5000
    data = np.random.normal(size=size, loc=2, scale=3)
    # data = np.concatenate([data, np.random.uniform(size=size * 1, low=-5, high=2.3)])
    # data = tf.random.poisson(shape=(13000,), lam=7, dtype=ztypes.float)
    limits = (-15, 5)
    kde = zfit.models.dist_tfp.GaussianKDE1DimExactV1(data=data, bandwidth=h, obs=zfit.Space("obs1", limits=limits))
    kde_adaptive = zfit.models.dist_tfp.GaussianKDE1DimExactV1(data=data, bandwidth='adaptiveV1',
                                                               obs=zfit.Space("obs1", limits=limits))
    kde_silverman = zfit.models.dist_tfp.GaussianKDE1DimExactV1(data=data, bandwidth='silverman',
                                                                obs=zfit.Space("obs1", limits=limits))

    integral = kde.integrate(limits=limits, norm_range=False)
    integral_adaptive = kde_adaptive.integrate(limits=limits, norm_range=False)
    integral_silverman = kde_silverman.integrate(limits=limits, norm_range=False)

    import matplotlib.pyplot as plt
    # plt.plot(data, kde_adaptive.pdf(data), 'x')

    data_plot = np.linspace(np.min(data) - 1, np.max(data) + 1, 1000)
    plt.hist(data, bins=20, density=1)
    plt.plot(data_plot, kde_adaptive.pdf(data_plot), 'x')
    # plt.plot(data_plot, kde_silverman.pdf(data_plot), 'x')
    # plt.plot(data_plot, kde.pdf(data_plot), 'x')
    plt.show()

    rel_tol = 0.04
    assert zfit.run(integral) == pytest.approx(expected_integral, rel=rel_tol)
    assert zfit.run(integral) == pytest.approx(expected_integral, rel=rel_tol)
    assert zfit.run(integral_adaptive) == pytest.approx(expected_integral, rel=rel_tol)
    assert zfit.run(integral_silverman) == pytest.approx(expected_integral, rel=rel_tol)

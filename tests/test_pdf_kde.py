#  Copyright (c) 2020 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
import zfit.models.dist_tfp
import zfit.models.kde
from zfit import ztypes


def test_simple_kde():
    expected_integral = 5 / 6
    h = zfit.Parameter("h", 0.9)

    size = 5000
    data = np.random.normal(size=size, loc=2, scale=3)

    limits = (-15, 5)
    obs = zfit.Space("obs1", limits=limits)
    data_truncated = obs.filter(data)
    # data = np.concatenate([data, np.random.uniform(size=size * 1, low=-5, high=2.3)])
    # data = tf.random.poisson(shape=(13000,), lam=7, dtype=ztypes.float)
    kde = zfit.models.kde.GaussianKDE1DimV1(data=data, bandwidth=h, obs=obs,
                                            truncate=False)
    kde_adaptive = zfit.models.kde.GaussianKDE1DimV1(data=data, bandwidth='adaptiveV1',
                                                     obs=obs,
                                                     truncate=False)
    kde_silverman = zfit.models.kde.GaussianKDE1DimV1(data=data, bandwidth='silverman',
                                                      obs=obs,
                                                      truncate=False)
    kde_adaptive_trunc = zfit.models.kde.GaussianKDE1DimV1(data=data_truncated, bandwidth='adaptiveV1',
                                                           obs=obs,
                                                           truncate=True)

    integral = kde.integrate(limits=limits, norm_range=False)
    integral_trunc = kde_adaptive_trunc.integrate(limits=limits, norm_range=False)
    integral_adaptive = kde_adaptive.integrate(limits=limits, norm_range=False)
    integral_silverman = kde_silverman.integrate(limits=limits, norm_range=False)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(data, kde_adaptive.pdf(data), 'x')
    #
    # data_plot = np.linspace(np.min(data) - 1, np.max(data) + 1, 1000)
    # plt.hist(data, bins=20, density=1)
    # plt.plot(data_plot, kde_adaptive.pdf(data_plot), 'x')
    # plt.plot(data_plot, kde_silverman.pdf(data_plot), 'x')
    # plt.plot(data_plot, kde.pdf(data_plot), 'x')
    # plt.show()

    rel_tol = 0.04
    assert zfit.run(integral_trunc) == pytest.approx(1., rel=rel_tol)
    assert zfit.run(integral) == pytest.approx(expected_integral, rel=rel_tol)
    assert zfit.run(integral_adaptive) == pytest.approx(expected_integral, rel=rel_tol)
    assert zfit.run(integral_silverman) == pytest.approx(expected_integral, rel=rel_tol)

    sample = kde_adaptive.sample(1000)
    sample2 = kde_adaptive_trunc.sample(1000)
    assert sample.nevents == 1000
    assert sample2.nevents == 1000

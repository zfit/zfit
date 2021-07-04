#  Copyright (c) 2021 zfit

import numpy as np
import pytest

import zfit
from zfit import z


@pytest.mark.skip()  # copy not yet implemented
def test_copy_kde():
    size = 500
    data = np.random.normal(size=size, loc=2, scale=3)

    limits = (-15, 5)
    obs = zfit.Space("obs1", limits=limits)
    kde_adaptive = zfit.pdf.GaussianKDE1DimV1(data=data, bandwidth='adaptive',
                                              obs=obs,
                                              truncate=False)
    kde_adaptive.copy()


def create_kde(kdetype, npoints=5000):
    import tensorflow as tf
    import zfit

    limits = (-13, 11)
    obs = zfit.Space("obs1", limits=limits)
    cb = zfit.pdf.CrystalBall(mu=2, sigma=3, alpha=1, n=25, obs=obs)
    gauss = zfit.pdf.Gauss(mu=-5, sigma=2.5, obs=obs)
    pdf = zfit.pdf.SumPDF([cb, gauss], fracs=0.8)
    data = pdf.sample(n=npoints)
    if kdetype == 0:
        with tf.init_scope():
            h = zfit.Parameter("h", 0.9)
        kde = zfit.pdf.GaussianKDE1DimV1(data=data, bandwidth=h, obs=obs,
                                         truncate=False)
    elif kdetype == 1:
        kde = zfit.pdf.GaussianKDE1DimV1(data=data, bandwidth='adaptive',
                                         obs=obs,
                                         truncate=False)
    elif kdetype == 2:
        kde = zfit.pdf.GaussianKDE1DimV1(data=data, bandwidth='silverman',
                                         obs=obs,
                                         truncate=False)
    elif kdetype == 3:
        data_truncated = obs.filter(data)[:, 0]  # TODO: fails if shape (n, 1)
        kde = zfit.pdf.GaussianKDE1DimV1(data=data_truncated, bandwidth='adaptive',
                                         obs=obs,
                                         truncate=True)
    elif kdetype == 4:
        kde = zfit.pdf.GaussianKDE1DimV1(data=data, bandwidth='isj',
                                         obs=obs,
                                         truncate=False)
    elif kdetype == 5:
        with tf.init_scope():
            h = zfit.Parameter("h", 0.9)

        kde = zfit.pdf.KDE1DimV1(data=data, obs=obs, bandwidth=h, use_grid=True)
    elif kdetype == 6:
        kde = zfit.pdf.KDE1DimFFTV1(data=data, obs=obs, bandwidth=0.8, num_grid_points=1000)
    elif kdetype == 7:
        kde = zfit.pdf.KDE1DimISJV1(data=data, obs=obs, num_grid_points=1000)
    else:
        raise ValueError(f'KDE type {kdetype} invalid.')
    return kde, pdf


# @pytest.mark.flaky(3)
@pytest.mark.parametrize('jit', [
    False,
    True
])
@pytest.mark.parametrize('kdetype', [(i, 5000) for i in range(8)] + [(i, 5_000_000) for i in range(6, 8)])
def test_simple_kde(kdetype, jit):
    import zfit
    if jit:
        run_jit = z.function(run)
        expected_integral, integral, name, prob, prob_true, rel_tol, sample, sample2, x = run_jit(kdetype)
    else:

        expected_integral, integral, name, prob, prob_true, rel_tol, sample, sample2, x = run(kdetype)

    expected_integral = zfit.run(expected_integral)

    assert zfit.run(integral) == pytest.approx(expected_integral, rel=rel_tol)
    assert tuple(sample.shape) == (1, 1)
    assert tuple(sample2.shape) == (1500, 1)
    assert prob.shape.rank == 1
    assert np.mean(prob - prob_true) < 0.07
    # make sure that on average, most values are close
    assert np.mean((prob / prob_true)[prob_true > np.mean(prob_true)] ** 2) == pytest.approx(1, abs=0.05)
    rtol = 0.05
    np.testing.assert_allclose(prob, prob_true, rtol=rtol, atol=0.01)

    import matplotlib.pyplot as plt
    plt.plot(x, prob, label=f'kde: {name} {kdetype}')
    plt.plot(x, prob_true, label='pdf')
    plt.legend()
    plt.show()


def run(kdetype):
    from zfit.z import numpy as znp
    kde, pdf = create_kde(*kdetype)
    integral = kde.integrate(limits=kde.space, norm_range=(-3, 2))
    expected_integral = kde.integrate(limits=kde.space, norm_range=(-3, 2))
    rel_tol = 0.04
    sample = kde.sample(1).value()
    sample2 = kde.sample(1500).value()
    x = znp.linspace(*kde.space.limit1d, 30000)
    prob = kde.pdf(x)
    prob_true = pdf.pdf(x)
    return expected_integral, integral, kde.name, prob, prob_true, rel_tol, sample, sample2, x

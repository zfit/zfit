"""Example test for a pdf or function"""
#  Copyright (c) 2020 zfit

import numpy as np
import pytest
import scipy.signal
import scipy.stats
import tensorflow as tf

import zfit.models.convolution
from zfit import z
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester

param1_true = 0.3
param2_true = 1.2


class FFTConvPDFV1NoSampling(zfit.pdf.FFTConvPDFV1):

    @zfit.supports()
    def _sample(self, n, limits):
        raise zfit.exception.SpecificFunctionNotImplementedError


@pytest.mark.parametrize('interpolation', (
    'linear',
    'spline',
    'spline:5',
    'spline:3'
))
def test_conv_simple(interpolation):
    n_points = 2432
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter('param1', -3)
    param2 = zfit.Parameter('param2', 0.3)
    gauss = zfit.pdf.Gauss(0., param2, obs=obs)
    func1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    func2 = zfit.pdf.Uniform(-1.2, -1, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    conv = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss, n=400, interpolation=interpolation)
    if interpolation == 'spline:5':
        assert conv._spline_order == 5
    elif interpolation == 'spline:3':
        assert conv._spline_order == 3

    x = tf.linspace(-5., 5., n_points)
    probs = conv.pdf(x=x)

    # true convolution
    true_conv = true_conv_np(func, gauss, obs, x)

    integral = conv.integrate(limits=obs)
    probs_np = probs.numpy()
    np.testing.assert_allclose(probs, true_conv, rtol=0.1, atol=0.01)

    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    assert len(probs_np) == n_points

    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, probs_np, label='zfit')
    # plt.plot(x, true_conv, label='numpy')
    # plt.legend()
    # plt.title(interpolation)
    # plt.show()


def test_onedim_sampling():
    obs_kernel = zfit.Space("obs1", limits=(-3, 3))
    obs = zfit.Space("obs1", limits=(5, 15))
    param2 = zfit.Parameter('param2', 0.3)
    # gauss = zfit.pdf.Gauss(0., param2, obs=obs)
    gauss = zfit.pdf.CrystalBall(0, param2, 0.5, 3, obs=obs_kernel)
    func1 = zfit.pdf.Uniform(8, 12, obs=obs)
    func2 = zfit.pdf.Uniform(11, 11.5, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    conv = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss)

    conv_nosample = FFTConvPDFV1NoSampling(func=func, kernel=gauss, n=1000)
    npoints_sample = 10000
    sample = conv.sample(npoints_sample)
    sample_nosample = conv_nosample.sample(npoints_sample)
    x = z.unstack_x(sample)
    xns = z.unstack_x(sample_nosample)
    assert scipy.stats.ks_2samp(x, xns).pvalue > 1e-10  # can vary a lot, but still means close

    # import matplotlib.pyplot as plt
    # plt.figure()
    # nbins = 50
    # _, bins, _ = plt.hist(x.numpy(), bins=nbins, label='custom', alpha=0.5)
    # plt.hist(xns.numpy(), bins=bins, label='fallback', alpha=0.5)
    # lower, upper = np.min(bins), np.max(bins)
    # linspace = tf.linspace(lower, upper, 1000)
    # plt.plot(linspace, conv.pdf(linspace) * npoints_sample / nbins * (upper - lower))
    # plt.legend()
    # plt.show()


def true_conv_np(func, gauss1, obs, x):
    # n_points = x.shape[0]
    # y_kernel = gauss1.pdf(x[n_points // 4:-n_points // 4])
    y_kernel = gauss1.pdf(x)
    y_func = func.pdf(x)
    true_conv = scipy.signal.fftconvolve(y_func, y_kernel, mode='same')
    # true_conv = true_conv[n_points // 4:  n_points * 5 // 4]
    true_conv /= np.mean(true_conv) * obs.rect_area()
    return true_conv

def true_conv_2d_np(func, gauss1, obs, x):
    # n_points = x.shape[0]
    # y_kernel = gauss1.pdf(x[n_points // 4:-n_points // 4])
    y_kernel = gauss1.pdf(x)
    y_func = func.pdf(x)
    n = int(np.sqrt(x.shape[0]))
    y_kernel = tf.reshape(y_kernel, (n, n))
    y_func = tf.reshape(y_func, (n, n))
    true_conv = scipy.signal.convolve(y_func, y_kernel, mode='same')
    # true_conv = true_conv[n_points // 4:  n_points * 5 // 4]
    true_conv /= np.mean(true_conv) * obs.rect_area()
    return true_conv


# @pytest.mark.skip  # not yet implemented
def test_conv_2D_simple():
    zfit.run.set_graph_mode(False)  # TODO: remove, just for debugging
    n_points = 200
    obs1 = zfit.Space("obs1", limits=(-5, 5))
    obs2 = zfit.Space("obs2", limits=(-6, 6))

    param2 = zfit.Parameter('param2', 0.4)
    gauss1 = zfit.pdf.Gauss(0., 0.5, obs=obs1)
    gauss22 = zfit.pdf.CrystalBall(0.0, param2, -0.2, 3, obs=obs2)

    obs1func = zfit.Space("obs1", limits=(4, 10))
    obs2func = zfit.Space("obs2", limits=(-6, 4))
    obs_func = obs1func * obs2func

    gauss21 = zfit.pdf.Gauss(-0.5, param2, obs=obs2func)
    func1 = zfit.pdf.Uniform(5, 8, obs=obs1func)
    func2 = zfit.pdf.Uniform(6, 7, obs=obs1func)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    func = func * gauss21
    gauss = gauss1 * gauss22
    conv = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss)

    start = obs_func.rect_lower
    stop = obs_func.rect_upper
    x_tensor = tf.random.uniform((n_points, 2), start, stop)
    x_tensor = tf.reshape(x_tensor, (-1, 2))
    linspace = tf.linspace(start, stop, num=n_points)
    linspace = tf.transpose(tf.meshgrid(*tf.unstack(linspace, axis=-1)))
    linspace = tf.reshape(linspace, (-1, 2))
    obs2d = obs1 * obs2
    x = zfit.Data.from_tensor(obs=obs_func, tensor=x_tensor)
    linspace_data = zfit.Data.from_tensor(obs=obs_func, tensor=linspace)
    probs_rnd = conv.pdf(x=x)
    probs = conv.pdf(x=linspace_data)

    # Numpy doesn't support ndim convolution?
    true_probs = true_conv_2d_np(func, gauss, obs=obs2d, x=linspace)
    # np.testing.assert_allclose(probs, true_probs, rtol=0.2, atol=0.1)
    integral = conv.integrate(limits=obs_func)
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    probs_np = probs_rnd.numpy()
    assert len(probs_np) == n_points
    # probs_plot = np.reshape(probs_np, (-1, n_points))
    # x_plot = linspace[0]
    # probs_plot_projx = np.sum(probs_plot, axis=0)
    # plt.plot(x_plot, probs_np)
    # probs_plot = np.reshape(probs_np, (n_points, n_points))
    # plt.imshow(probs_plot)
    # plt.show()

    # test the sampling
    conv_nosample = FFTConvPDFV1NoSampling(func=func, kernel=gauss)

    npoints_sample = 10000
    sample = conv.sample(npoints_sample)
    sample_nosample = conv_nosample.sample(npoints_sample)
    x, y = z.unstack_x(sample)
    xns, yns = z.unstack_x(sample_nosample)
    import matplotlib.pyplot as plt
    true_probsr = tf.reshape(true_probs, (n_points, n_points))
    probsr = tf.reshape(probs, (n_points, n_points))
    plt.figure()
    # plt.hist(x, bins=50, label='custom', alpha=0.5)
    # plt.hist(xns, bins=50, label='fallback', alpha=0.5)
    # plt.legend()
    # plt.show()

"""Example test for a pdf or function."""
#  Copyright (c) 2023 zfit

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal
import scipy.stats
import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit.util.exception import WorkInProgressError

param1_true = 0.3
param2_true = 1.2


class FFTConvPDFV1NoSampling(zfit.pdf.FFTConvPDFV1):
    @zfit.supports()
    def _sample(self, n, limits):
        raise zfit.exception.SpecificFunctionNotImplemented


interpolation_methods = ("linear", "spline", "spline:5", "spline:3")


@pytest.mark.parametrize("interpolation", interpolation_methods)
def test_conv_simple(interpolation):
    n_points = 2432
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter("param1", -3)
    param2 = zfit.Parameter("param2", 0.3)
    gauss = zfit.pdf.Gauss(0.0, param2, obs=obs)
    func1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    func2 = zfit.pdf.Uniform(-1.2, -1, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    conv = zfit.pdf.FFTConvPDFV1(
        func=func, kernel=gauss, n=100, interpolation=interpolation
    )
    if interpolation == "spline:5":
        assert conv._conv_spline_order == 5
    elif interpolation == "spline:3":
        assert conv._conv_spline_order == 3

    x = tf.linspace(-5.0, 5.0, n_points)
    probs = conv.pdf(x=x)

    # true convolution
    true_conv = true_conv_np(
        func, gauss, obs, x, xkernel=tf.linspace(*obs.limit1d, num=n_points)
    )

    integral = conv.integrate(limits=obs)
    probs_np = probs.numpy()
    np.testing.assert_allclose(probs, true_conv, rtol=0.01, atol=0.01)

    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    assert len(probs_np) == n_points

    plt.figure()
    plt.title(f"Conv FFT 1Dim, interpolation={interpolation}")
    plt.plot(x, probs_np, label="zfit")
    plt.plot(x, true_conv, label="numpy")
    plt.legend()
    # pytest.zfit_savefig()


@pytest.mark.parametrize("interpolation", interpolation_methods)
def test_asymetric_limits(interpolation):
    from numpy import linspace

    import zfit
    from zfit.models.convolution import FFTConvPDFV1

    ## Space
    low_obs = -30
    high_obs = 30
    obs = zfit.Space("space", limits=[low_obs, high_obs])

    ## PDFs
    uniform1 = zfit.pdf.Uniform(low=-10, high=10, obs=obs)
    uniform2 = zfit.pdf.Uniform(low=-10, high=10, obs=obs)

    conv_uniforms_1 = FFTConvPDFV1(
        func=uniform1,
        kernel=uniform2,
        limits_kernel=(-18, 18),
        interpolation=interpolation,
    )

    conv_uniforms_2 = FFTConvPDFV1(
        func=uniform1,
        kernel=uniform2,
        limits_kernel=(-12, 12),
        interpolation=interpolation,
    )
    conv_uniforms_3 = FFTConvPDFV1(
        func=uniform1,
        kernel=uniform2,
        limits_kernel=(-25, 12),
        interpolation=interpolation,
    )

    x = linspace(low_obs, high_obs, 300)

    tol = 5e-3
    # If this fails, we're too sensitive
    np.testing.assert_allclose(
        conv_uniforms_1.pdf(x), conv_uniforms_2.pdf(x), rtol=tol, atol=tol
    )

    # this is the "actual" test
    np.testing.assert_allclose(
        conv_uniforms_1.pdf(x), conv_uniforms_3.pdf(x), rtol=tol, atol=tol
    )


@pytest.mark.parametrize("interpolation", interpolation_methods)
def test_conv_1d_shifted(interpolation):
    kerlim = (-3, 3)  # symmetric to make the np conv comparison simple
    obs_kernel = zfit.Space("obs1", limits=kerlim)
    obs = zfit.Space("obs1", limits=(5, 15))
    func1 = zfit.pdf.GaussianKDE1DimV1(obs=obs, data=np.random.uniform(6, 12, size=100))
    # func1 = zfit.pdf.Uniform(6, 12, obs=obs)
    func2 = zfit.pdf.Uniform(11, 11.5, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)

    func1k = zfit.pdf.Gauss(0.0, 1, obs=obs_kernel)
    func2k = zfit.pdf.Gauss(1.0, 0.4, obs=obs_kernel)
    funck = zfit.pdf.SumPDF([func1k, func2k], 0.5)

    conv = zfit.pdf.FFTConvPDFV1(func=func, kernel=funck, n=200)

    xnp = znp.linspace(obs_kernel.rect_lower, obs.rect_upper, 4023)

    # true convolution
    kernel_points = obs_kernel.filter(xnp)
    x = obs.filter(xnp)
    probs = conv.pdf(x=x)
    true_conv = true_conv_np(func, funck, obs, x=x, xkernel=kernel_points)

    integral = conv.integrate(
        limits=obs,
    )
    probs_np = probs.numpy()
    np.testing.assert_allclose(probs_np, true_conv, rtol=0.01, atol=0.01)

    assert pytest.approx(1, rel=1e-3) == integral.numpy()

    plt.figure()
    plt.title("Conv FFT 1Dim shift testing")
    plt.plot(x, probs_np, label="zfit")
    plt.plot(x, true_conv, label="numpy")
    plt.legend()
    pytest.zfit_savefig()


@pytest.mark.parametrize("interpolation", interpolation_methods)
@pytest.mark.flaky(reruns=3)
def test_onedim_sampling(interpolation):
    # there is a sampling shortcut, so we test if it also works without the shortcut
    obs_kernel = zfit.Space("obs1", limits=(-3, 1))
    obs = zfit.Space("obs1", limits=(5, 15))
    func1 = zfit.pdf.Uniform(6, 12, obs=obs)
    func2 = zfit.pdf.Uniform(11, 11.5, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)

    func1k = zfit.pdf.Uniform(-2, 1, obs=obs_kernel)
    func2k = zfit.pdf.Uniform(-0.5, 1.0, obs=obs_kernel)
    funck = zfit.pdf.SumPDF([func1k, func2k], 0.5)
    conv = zfit.pdf.FFTConvPDFV1(
        func=func, kernel=funck, n=200, interpolation=interpolation
    )

    conv_nosample = FFTConvPDFV1NoSampling(
        func=func, kernel=funck, n=200, interpolation=interpolation
    )
    npoints_sample = 10000
    sample = conv.sample(npoints_sample)
    sample_nosample = conv_nosample.sample(npoints_sample)
    x = z.unstack_x(sample)
    xns = z.unstack_x(sample_nosample)
    assert (
        scipy.stats.ks_2samp(x, xns).pvalue > 1e-3
    )  # can vary a lot, but still means close


def true_conv_np(func, gauss1, obs, x, xkernel):
    y_kernel = gauss1.pdf(xkernel)
    y_func = func.pdf(x)
    true_conv = scipy.signal.fftconvolve(y_func, y_kernel, mode="same")
    true_conv /= np.mean(true_conv) * obs.rect_area()
    return true_conv


def true_conv_2d_np(func, gauss1, obsfunc, xfunc, xkernel):
    y_func = func.pdf(xfunc)
    y_kernel = gauss1.pdf(xkernel)
    nfunc = int(np.sqrt(xfunc.shape[0]))
    y_func = tf.reshape(y_func, (nfunc, nfunc))
    nkernel = int(np.sqrt(xkernel.shape[0]))
    y_kernel = tf.reshape(y_kernel, (nkernel, nkernel))
    true_conv = scipy.signal.convolve(y_func, y_kernel, mode="same")
    true_conv /= np.mean(true_conv) * obsfunc.rect_area()
    return tf.reshape(true_conv, xfunc.shape[0])


def test_max_1dim():
    obs1 = zfit.Space("obs1", limits=(-2, 4))
    obs2 = zfit.Space("obs2", limits=(-6, 4))

    param2 = zfit.Parameter("param2", 0.4)
    gauss1 = zfit.pdf.Gauss(1.0, 0.5, obs=obs1)
    gauss22 = zfit.pdf.CrystalBall(0.0, param2, -0.2, 3, obs=obs2)

    obs1func = zfit.Space("obs1", limits=(4, 10))
    obs2func = zfit.Space("obs2", limits=(-6, 4))

    gauss21 = zfit.pdf.Gauss(-0.5, param2, obs=obs2func)
    func1 = zfit.pdf.Uniform(5, 8, obs=obs1func)
    func2 = zfit.pdf.Uniform(6, 7, obs=obs1func)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    func = func * gauss21
    gauss = gauss1 * gauss22
    with pytest.raises(WorkInProgressError):
        _ = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss)


@pytest.mark.skip  # not yet implemented WIP
def test_conv_2D_simple():
    # zfit.run.set_graph_mode(False)  # TODO: remove, just for debugging
    # raise WorkInProgressError("2D convolution not yet implemented, re-activate if so")
    n_points = 1000
    # obs1 = zfit.Space("obs1", limits=(-2, 4))
    # obs2 = zfit.Space("obs2", limits=(-6, 4))
    obs1 = zfit.Space("obs1", limits=(-5, 5))
    obs2 = zfit.Space("obs2", limits=(-5, 5))
    obskernel = obs1 * obs2

    param2 = zfit.Parameter("param2", 0.4)
    gauss1 = zfit.pdf.Gauss(1.0, 0.5, obs=obs1)
    gauss22 = zfit.pdf.CrystalBall(0.0, param2, -0.2, 3, obs=obs2)

    gauss1 = zfit.pdf.Uniform(-1, 1, obs=obs1)
    gauss22 = zfit.pdf.Uniform(-2, 2, obs=obs2)

    obs1func = zfit.Space("obs1", limits=(-10, 10))
    obs2func = zfit.Space("obs2", limits=(-26, 26))
    obs_func = obs1func * obs2func

    gauss21 = zfit.pdf.Gauss(-0.5, param2, obs=obs2func)
    func1 = zfit.pdf.Uniform(2, 8, obs=obs1func)
    func2 = zfit.pdf.Uniform(6, 7, obs=obs1func)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    func = func * gauss21
    gauss = gauss1 * gauss22
    conv = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss)

    start = obs_func.rect_lower
    stop = obs_func.rect_upper
    x_tensor = z.random.uniform((n_points, 2), start, stop)
    x_tensor = tf.reshape(x_tensor, (-1, 2))
    linspace = tf.linspace(start, stop, num=n_points)
    linspace = tf.transpose(tf.meshgrid(*tf.unstack(linspace, axis=-1)))
    linspace_func = tf.reshape(linspace, (-1, 2))

    # linspace_full = tf.linspace((-8, -8), (12, 12), num=n_points)
    # linspace_full = tf.transpose(tf.meshgrid(*tf.unstack(linspace_full, axis=-1)))
    # linspace_full = tf.reshape(linspace_full, (-1, 2))

    linspace_kernel = tf.linspace(
        obskernel.rect_lower, obskernel.rect_upper, num=n_points
    )
    linspace_kernel = tf.transpose(tf.meshgrid(*tf.unstack(linspace_kernel, axis=-1)))
    linspace_kernel = tf.reshape(linspace_kernel, (-1, 2))
    # linspace_kernel = obskernel.filter(linspace_full)
    # linspace_func = obs_func.filter(linspace_full)

    x = zfit.Data.from_tensor(obs=obs_func, tensor=x_tensor)
    linspace_data = zfit.Data.from_tensor(obs=obs_func, tensor=linspace)
    probs_rnd = conv.pdf(x=x)
    probs = conv.pdf(x=linspace_data)

    # Numpy doesn't support ndim convolution?
    true_probs = true_conv_2d_np(
        func, gauss, obsfunc=obs_func, xfunc=linspace_func, xkernel=linspace_kernel
    )
    import matplotlib.pyplot as plt

    # np.testing.assert_allclose(probs, true_probs, rtol=0.2, atol=0.1)
    integral = conv.integrate(
        limits=obs_func,
    )
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    probs_np = probs_rnd.numpy()
    assert len(probs_np) == n_points
    # probs_plot = np.reshape(probs_np, (-1, n_points))
    # x_plot = linspace[0:, ]
    # probs_plot_projx = np.sum(probs_plot, axis=0)
    # plt.plot(x_plot, probs_np)
    # probs_plot = np.reshape(probs_np, (n_points, n_points))
    # plt.imshow(probs_plot)
    # plt.show()

    true_probsr = tf.reshape(true_probs, (n_points, n_points))
    probsr = tf.reshape(probs, (n_points, n_points))
    plt.figure()
    plt.imshow(true_probsr, label="true probs")
    plt.title("true probs")

    plt.figure()
    plt.imshow(probsr, label="zfit conv")
    plt.title("zfit conv")

    # test the sampling
    conv_nosample = FFTConvPDFV1NoSampling(func=func, kernel=gauss)

    npoints_sample = 10000
    sample = conv.sample(npoints_sample)
    sample_nosample = conv_nosample.sample(npoints_sample)
    x, y = z.unstack_x(sample)
    xns, yns = z.unstack_x(sample_nosample)

    plt.figure()
    plt.title("FFT conv, custom sampling, addition")
    plt.hist2d(x, y, bins=30)
    # pytest.zfit_savefig()

    plt.figure()
    plt.title("FFT conv, fallback sampling, accept-reject")
    plt.hist2d(xns, yns, bins=30)
    # pytest.zfit_savefig()

    plt.figure()
    plt.title("FFT conv x projection")
    plt.hist(x.numpy(), bins=50, label="custom", alpha=0.5)
    plt.hist(xns.numpy(), bins=50, label="fallback", alpha=0.5)
    plt.legend()
    # pytest.zfit_savefig()

    plt.figure()
    plt.title("FFT conv y projection")
    plt.hist(y.numpy(), bins=50, label="custom", alpha=0.5)
    plt.hist(yns.numpy(), bins=50, label="fallback", alpha=0.5)
    plt.legend()
    # pytest.zfit_savefig()

"""Example test for a pdf or function"""
#  Copyright (c) 2020 zfit

import pytest
import tensorflow as tf

import zfit.models.convolution
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester

# Important, do the imports below

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


@pytest.mark.parametrize('interpolation', (
    'linear',
    'spline',
))
def test_conv_simple(interpolation):
    # test special properties  here
    n_points = 200
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter('param1', -3)
    param2 = zfit.Parameter('param2', 0.3)
    gauss1 = zfit.pdf.Gauss(0., param2, obs=obs)
    func1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    func2 = zfit.pdf.Uniform(-1.2, -1, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    # func = zfit.pdf.(-0.1, obs=obs)
    conv = zfit.pdf.FFTConv1DV1(func=func,
                                kernel=gauss1,
                                interpolation=interpolation,
                                npoints=100,
                                )

    x = tf.linspace(-5., 5., n_points)
    probs = conv.pdf(x=x)
    # probs = func.pdf(x=x)
    integral = conv.integrate(limits=obs)
    probs_np = probs.numpy()
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    assert len(probs_np) == n_points
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, probs_np)
    plt.title(interpolation)
    plt.show()


def test_conv_2D_simple():
    # test special properties  here
    zfit.run.set_graph_mode(False)
    n_points = 200
    obs1 = zfit.Space("obs1", limits=(-5, 5))
    obs2 = zfit.Space("obs2", limits=(-6, 8))
    param1 = zfit.Parameter('param1', -3)
    param2 = zfit.Parameter('param2', 0.4)
    gauss1 = zfit.pdf.Gauss(0., param2, obs=obs1)
    gauss21 = zfit.pdf.Gauss(0.5, param2, obs=obs2)
    gauss22 = zfit.pdf.Gauss(0.3, param2 + 0.7, obs=obs2)
    func1 = zfit.pdf.Uniform(param1, param2, obs=obs1)
    func2 = zfit.pdf.Uniform(-1.2, -1, obs=obs1)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    func = func * gauss21
    gauss = gauss1 * gauss22
    # func = zfit.pdf.(-0.1, obs=obs)
    conv = zfit.pdf.FFTConv1DV1(func=func,
                                kernel=gauss,
                                # limits_kernel=(-1, 1)
                                )

    # x = tf.linspace((-5., -6), (5., 6), n_points)
    x_tensor = tf.random.uniform((n_points, 2), (-5., -6), (5., 6))
    # start = (-5., -6.)
    # stop = (5., 6.)
    # linspace = tf.transpose(tf.linspace(start, stop, num=n_points))
    # x_tensor = tf.transpose(tf.meshgrid(*linspace))
    x_tensor = tf.reshape(x_tensor, (-1, 2))
    obs2d = obs1 * obs2
    x = zfit.Data.from_tensor(obs=obs2d, tensor=x_tensor)
    probs = conv.pdf(x=x)
    # probs = func.pdf(x=x)
    integral = conv.integrate(limits=obs2d)
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    probs_np = probs.numpy()
    assert len(probs_np) == n_points
    # probs_plot = np.reshape(probs_np, (-1, n_points))
    # x_plot = linspace[0]
    # probs_plot_projx = np.sum(probs_plot, axis=0)
    # plt.plot(x_plot, probs_np)
    # probs_plot = np.reshape(probs_np, (n_points, n_points))
    # plt.imshow(probs_plot)
    # plt.show()
    # assert len(conv.get_dependents(only_floating=False)) == 2  # TODO: activate again with fixed params

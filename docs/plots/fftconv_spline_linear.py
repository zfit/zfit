#  Copyright (c) 2020 zfit

import matplotlib.pyplot as plt
import tensorflow as tf

import zfit


def plot_conv_comparison():
    # test special properties  here
    n_point_plotting = 2000
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter('param1', -3)
    param2 = zfit.Parameter('param2', 0.3)
    gauss1 = zfit.pdf.Gauss(0., param2, obs=obs)
    func1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    func2 = zfit.pdf.Uniform(-1.2, -1, obs=obs)
    func = zfit.pdf.SumPDF([func1, func2], 0.5)
    n_points_conv = 50
    conv_lin = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss1, n=n_points_conv, interpolation='linear')
    conv_spline = zfit.pdf.FFTConvPDFV1(func=func, kernel=gauss1, n=n_points_conv, interpolation='spline')

    x = tf.linspace(-5., 3., n_point_plotting)
    probs_lin = conv_lin.pdf(x=x)
    probs_spline = conv_spline.pdf(x=x)

    plt.figure()
    plt.plot(x, probs_lin, label='linear')
    plt.plot(x, probs_spline, label='spline')
    plt.legend()
    plt.title(f"FFT Conv with interpolation: {n_points_conv} points")
    plt.show(block=True)


if __name__ == '__main__':
    plot_conv_comparison()

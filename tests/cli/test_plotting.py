#  Copyright (c) 2024 zfit
import pytest
from matplotlib import pyplot as plt

import zfit
from zfit.util import plotter


def test_plot_1d_simple():
    obs = zfit.Space("obs1", -1, 1)
    gauss = zfit.pdf.Gauss(mu=0.2, sigma=0.17, obs=obs)

    plotter.plot_model_pdf(gauss, obs=obs)
    plotter.plt.show()


def test_plot_sum_simple():
    obs = zfit.Space("obs1", -1, 1)
    gauss1 = zfit.pdf.Gauss(mu=0.2, sigma=0.17, obs=obs, label="Gauss1")
    gauss2 = zfit.pdf.Gauss(mu=-0.2, sigma=0.17, obs=obs, label="Gauss2")
    sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=0.5, label="Model")

    plotter.plot_comp_model_pdf(sum_pdf, obs=obs)
    plotter.plt.show()


def test_plot_3d_simple():
    obs1 = zfit.Space("obs1", 10, 100)
    obs2 = zfit.Space("obs2", -5, 4)
    obs3 = zfit.Space("obs3", 0, 1)
    gauss1 = zfit.pdf.Gauss(mu=49, sigma=4, obs=obs1)
    gauss2 = zfit.pdf.Gauss(mu=-3, sigma=0.9, obs=obs2)
    gauss3 = zfit.pdf.Gauss(mu=0.2, sigma=0.07, obs=obs3)
    model = zfit.pdf.ProductPDF([gauss1, gauss2, gauss3])

    with pytest.raises(ValueError):
        plotter.plot_model_pdf(model)
    obsplot = zfit.Space("obs2", -5, -3)
    plt.figure()
    plotter.plot_model_pdf(model, obs=obsplot)
    plt.show()
    plt.figure()
    plotter.plot_model_pdf(model, obs="obs2")
    plt.show()

    model_ext = model.create_extended(1000)
    with pytest.raises(ValueError):
        plotter.plot_model_pdf(model_ext)
    plotter.plot_model_pdf(model_ext, obs="obs2", extended=True)
    plt.show()

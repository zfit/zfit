#  Copyright (c) 2024 zfit
import pytest
from matplotlib import pyplot as plt

import zfit
from zfit.util import plotter

# todo: add actual assertions?

folder = "plotting"
def test_plot_1d_simple():
    obs = zfit.Space("obs1", -1, 1, label="$obs_1$ [GeV$^2$]")
    gauss = zfit.pdf.Gauss(mu=0.2, sigma=0.17, obs=obs)

    plt.figure()
    plt.title("Simple 1D plot using plot_model_pdf")
    plotter.plot_model_pdfV1(gauss, obs=obs)
    pytest.zfit_savefig(folder=folder)



def test_plot_sum_simple():
    obs = zfit.Space("obs1", -1, 1)
    gauss1 = zfit.pdf.Gauss(mu=0.2, sigma=0.17, obs=obs, label="Gauss1")
    gauss2 = zfit.pdf.Gauss(mu=-0.2, sigma=0.17, obs=obs, label="Gauss2")
    sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=0.5, label="Model")

    plt.figure()
    plt.title("Simple sum plot using plot_sumpdf_components_pdfV1")
    sum_pdf.plot.plotpdf()
    sum_pdf.plot.comp.plotpdf()
    pytest.zfit_savefig(folder=folder)



def test_plot_3d_simple():
    obs1 = zfit.Space("obs1", 10, 100)
    obs2 = zfit.Space("obs2", -5, 4)
    obs3 = zfit.Space("obs3", 0, 1)
    gauss1 = zfit.pdf.Gauss(mu=49, sigma=4, obs=obs1)
    gauss2 = zfit.pdf.Gauss(mu=-3, sigma=0.9, obs=obs2)
    gauss3 = zfit.pdf.Gauss(mu=0.2, sigma=0.07, obs=obs3)
    model = zfit.pdf.ProductPDF([gauss1, gauss2, gauss3])

    with pytest.raises(ValueError):
        plotter.plot_model_pdfV1(model)
    obsplot = zfit.Space("obs2", -5, -3)
    plt.figure()
    plt.title("Simple 3D plot using plot_model_pdf with space")
    plotter.plot_model_pdfV1(model, obs=obsplot)
    pytest.zfit_savefig(folder=folder)

    plt.figure()
    plt.title("Simple 3D plot using plot_model_pdf")
    plotter.plot_model_pdfV1(model, obs="obs2")
    pytest.zfit_savefig(folder=folder)

    model_ext = model.create_extended(1000)
    plt.figure()
    plt.title("Extended model 3D")
    with pytest.raises(ValueError):
        plotter.plot_model_pdfV1(model_ext)
    plotter.plot_model_pdfV1(model_ext, obs="obs2", extended=True)
    pytest.zfit_savefig(folder=folder)

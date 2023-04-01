#  Copyright (c) 2023 zfit

import pytest

try:
    import ROOT
except ImportError:
    pytest.skip("ROOT not installed", allow_module_level=True)

import zfit
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit


@pytest.fixture
def toy_data():
    # Generate toy data with RooFit
    x = ROOT.RooRealVar("x", "Mass", 0, 10)
    mean = ROOT.RooRealVar("mean", "mean", 5, 0, 10)
    sigma = ROOT.RooRealVar("sigma", "sigma", 1, 0, 5)
    fraction = ROOT.RooRealVar("fraction", "fraction", 0.5, 0, 1)
    gauss = ROOT.RooGaussian("gauss", "Gaussian", x, mean, sigma)
    expo = ROOT.RooExponential("expo", "Exponential", x, ROOT.RooFit.RooConst(-0.3))
    model = ROOT.RooAddPdf(
        "model", "Model", ROOT.RooArgList(gauss, expo), ROOT.RooArgList(fraction)
    )
    data = model.generate(ROOT.RooArgSet(x), 100000)
    return data


def test_roofit_vs_zfit(toy_data):
    # Perform fit with RooFit
    x = ROOT.RooRealVar("x", "Mass", 0, 10)
    mean = ROOT.RooRealVar("mean", "mean", 5, 0, 10)
    sigma = ROOT.RooRealVar("sigma", "sigma", 1, 0, 5)
    fraction = ROOT.RooRealVar("fraction", "fraction", 0.5, 0, 1)
    gauss = ROOT.RooGaussian("gauss", "Gaussian", x, mean, sigma)
    expo = ROOT.RooExponential("expo", "Exponential", x, ROOT.RooFit.RooConst(-0.3))
    model = ROOT.RooAddPdf(
        "model", "Model", ROOT.RooArgList(gauss, expo), ROOT.RooArgList(fraction)
    )
    roofit_result = model.fitTo(toy_data, ROOT.RooFit.Save(True), ROOT.RooFit.Hesse())
    # run hesse

    # Perform fit with zfit
    obs = zfit.Space("x", limits=(0, 10))
    mean = zfit.Parameter("mean", 5, 0, 10)
    sigma = zfit.Parameter("sigma", 1, 0, 5, step_size=0.01)
    fraction = zfit.Parameter("fraction", 0.5, 0, 1)
    gauss = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    expo = zfit.pdf.Exponential(obs=obs, lambda_=zfit.Parameter("lambda", -0.3))
    model = zfit.pdf.SumPDF([gauss, expo], fracs=fraction)
    toy_data_zfit = zfit.data.Data.from_numpy(obs=obs, array=toy_data.to_numpy()["x"])
    nll = UnbinnedNLL(model=model, data=toy_data_zfit, options={"subtr_const": False})
    minimizer = Minuit()

    zfit_result = minimizer.minimize(nll)
    zfit_result.hesse()

    # Compare uncertainties
    mean_unc = zfit_result.params[mean]["hesse"]["error"]
    assert (
        pytest.approx(roofit_result.floatParsFinal().find("mean").getError(), rel=0.05)
        == mean_unc
    )
    sigma_unc = zfit_result.params[sigma]["hesse"]["error"]
    assert (
        pytest.approx(roofit_result.floatParsFinal().find("sigma").getError(), rel=0.15)
        == sigma_unc
    )
    # Compare results
    assert pytest.approx(roofit_result.minNll(), rel=1e-3) == zfit_result.fmin
    assert (
        pytest.approx(
            roofit_result.floatParsFinal().find("mean").getVal(), abs=3 * mean_unc
        )
        == mean.value()
    )
    assert (
        pytest.approx(
            roofit_result.floatParsFinal().find("sigma").getVal(), abs=3 * sigma_unc
        )
        == sigma.value()
    )

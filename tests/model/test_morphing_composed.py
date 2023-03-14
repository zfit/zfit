#  Copyright (c) 2023 zfit

import pytest


def test_moprhing_sum():
    import numpy as np
    import zfit
    import zfit.z.numpy as znp  # numpy-like backend interface

    normal_np = np.random.normal(loc=2.0, scale=1.3, size=10000)
    obs = zfit.Space("x", limits=(-10, 10))
    true_mu = 1.0
    mu = zfit.Parameter("mu", true_mu, -4, 6)
    true_sigma = 1.0
    sigma = zfit.Parameter("sigma", true_sigma, 0.1, 10)
    model_nobin = zfit.pdf.Gauss(mu, sigma, obs)
    data_nobin = zfit.Data.from_numpy(obs, normal_np)
    minimizer = zfit.minimize.Minuit()
    # make binned
    nbins = 50
    data = data_nobin.to_binned(nbins)
    model = model_nobin.to_binned(data.space)
    obs_binned = data.space
    sig_true = 10_000
    sig_yield = zfit.Parameter("sig_yield", sig_true, 0, 100_000)
    template_hist = model.sample(
        25_000
    ).to_hist()  # let's assume we have 25_000 MC events in our template
    histpdf = zfit.pdf.HistogramPDF(template_hist, extended=sig_yield)
    sig_pdf = zfit.pdf.BinwiseScaleModifier(
        histpdf, modifiers=True
    )  # or we could give a list of parameters matching each bin
    modifiers = sig_pdf.params
    bkg_hist = zfit.Data.from_numpy(
        obs=obs, array=np.random.exponential(scale=20, size=100_000) - 10
    ).to_binned(obs_binned)
    bkg_hist_m1 = zfit.Data.from_numpy(
        obs=obs, array=np.random.exponential(scale=35, size=100_000) - 10
    ).to_binned(obs_binned)
    bkg_hist_m05 = zfit.Data.from_numpy(
        obs=obs, array=np.random.exponential(scale=26, size=100_000) - 10
    ).to_binned(obs_binned)
    bkg_hist_p1 = zfit.Data.from_numpy(
        obs=obs, array=np.random.exponential(scale=17, size=100_000) - 10
    ).to_binned(obs_binned)

    bkg_hists = {-1: bkg_hist_m1, -0.5: bkg_hist_m05, 0: bkg_hist, 1: bkg_hist_p1}
    bkg_histpdfs = {k: zfit.pdf.HistogramPDF(v) for k, v in bkg_hists.items()}
    alpha = zfit.Parameter("alpha", 0, -3, 3)
    bkg_true = 45_000
    bkg_yield = zfit.Parameter("bkg_yield", bkg_true)
    bkg_pdf = zfit.pdf.SplineMorphingPDF(alpha, bkg_histpdfs, extended=bkg_yield)

    model = zfit.pdf.BinnedSumPDF([sig_pdf, bkg_pdf])
    with zfit.param.set_values([alpha], [0.1]):
        data = model.sample()
    uncertainties = 1 / znp.maximum(template_hist.counts() ** 0.5, 1)
    modifier_constraints = zfit.constraint.GaussianConstraint(
        params=list(modifiers.values()),
        observation=np.ones(len(modifiers)),
        uncertainty=uncertainties,
    )
    alpha_constraint = zfit.constraint.GaussianConstraint(alpha, 0, 1)
    loss_binned = zfit.loss.ExtendedBinnedNLL(
        model, data, constraints=[modifier_constraints, alpha_constraint]
    )
    result = minimizer.minimize(loss_binned)
    assert result.valid
    params_to_test = [bkg_yield, alpha] + list(modifiers.values())[::7]

    result.hesse(name="hesse", params=params_to_test)
    result.errors(name="zfit", method="zfit_errors", params=params_to_test)
    result.errors(name="minos", method="minuit_minos", params=params_to_test)
    assert pytest.approx(sig_true, abs=3 * sig_true**0.5) == np.array(sig_yield)
    assert (
        pytest.approx(bkg_true, abs=3 * bkg_true**0.5)
        == result.params[bkg_yield]["value"]
    )
    assert pytest.approx(1, abs=0.003) == np.mean(list(modifiers.values()))
    for p in params_to_test:
        p = result.params[p]
        lower = p["zfit"]["lower"]
        upper = p["zfit"]["upper"]
        avg = (abs(upper) + abs(lower)) / 2
        assert pytest.approx(lower, rel=0.03) == p["minos"]["lower"]
        assert pytest.approx(upper, rel=0.03) == p["minos"]["upper"]
        assert pytest.approx(avg, rel=0.15) == p["hesse"]["error"]

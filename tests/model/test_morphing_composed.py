#  Copyright (c) 2022 zfit


def test_moprhing_sum():
    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np
    import zfit
    import zfit.z.numpy as znp  # numpy-like backend interface

    normal_np = np.random.normal(loc=2.0, scale=1.3, size=10000)
    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -4, 6)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    model_nobin = zfit.pdf.Gauss(mu, sigma, obs)
    data_nobin = zfit.Data.from_numpy(obs, normal_np)
    loss_nobin = zfit.loss.UnbinnedNLL(model_nobin, data_nobin)
    minimizer = zfit.minimize.Minuit()
    # make binned
    nbins = 50
    data = data_nobin.to_binned(nbins)
    model = model_nobin.to_binned(data.space)
    loss = zfit.loss.BinnedNLL(model, data)
    # %%
    result = minimizer.minimize(loss)
    print(result)
    # %%
    result.hesse(name="hesse")
    # %%
    result.errors(name="errors")
    # %%
    print(result)
    obs_binned_auto = data.space
    print(obs_binned_auto)
    # %%
    print(
        f"is_binned: {obs_binned_auto.is_binned}, binned obs binning: {obs_binned_auto.binning}"
    )
    print(f"is_binned: {obs.is_binned}, unbinned obs binning:{obs.binning}")
    # %% md
    # %%
    obs_binned = obs.with_binning(nbins)
    print(obs_binned)
    # or we can create binnings
    binning_regular = zfit.binned.RegularBinning(nbins, -10, 10, name="x")
    binning_variable = zfit.binned.VariableBinning(
        [-10, -6, -1, -0.1, 0.4, 3, 10], name="x"
    )
    obs_binned_variable = zfit.Space(binning=binning_variable)
    print(obs_binned_variable, obs_binned_variable.binning)
    # %% md
    ## Converting data, models
    # %%
    data_nobin.to_binned(obs_binned_variable)
    # %%
    model_nobin.to_binned(obs_binned_variable)
    template_hist = model.to_hist()
    h_scaled = template_hist * [10_000]
    data.values()
    binned_sample = model.sample(n=1_000)
    # %% md
    plt.figure()
    plt.title("Counts plot")
    mplhep.histplot(data, label="data")
    mplhep.histplot(
        model.to_hist() * [data.nevents], label="model"
    )  # scaling up since model is not extended, i.e. has no yield
    plt.legend()
    # %%
    plt.figure()
    plt.title("Counts plot")
    mplhep.histplot(binned_sample, label="sampled data")
    mplhep.histplot(
        model.to_hist() * [binned_sample.nevents], label="model"
    )  # scaling up since model is not extended, i.e. has no yield
    plt.legend()
    # %%
    # or using unbinned data points, we can do a density plot
    plt.figure()
    plt.title("Density plot")
    mplhep.histplot(data.to_hist(), density=True, label="data")
    x = znp.linspace(-10, 10, 200)
    plt.plot(x, model.pdf(x), label="model")
    plt.legend()
    print(zfit.loss.__all__)
    histpdf = zfit.pdf.HistogramPDF(h_scaled)  # fixed yield
    print(np.sum(histpdf.counts()))
    # %%
    sig_yield = zfit.Parameter("sig_yield", 4_000, 0, 100_000)
    # %%
    template_hist = model.sample(
        25_000
    ).to_hist()  # let's assume we have 25_000 MC events in our template
    histpdf = zfit.pdf.HistogramPDF(template_hist, extended=sig_yield)
    print(np.sum(histpdf.counts()))
    histpdf.space.binning.size
    # %%
    sig_pdf = zfit.pdf.BinwiseScaleModifier(
        histpdf, modifiers=True
    )  # or we could give a list of parameters matching each bin
    modifiers = sig_pdf.params
    sig_pdf.get_yield()
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
    plt.figure()
    mplhep.histplot(list(bkg_hists.values()), label=list(bkg_hists.keys()))
    plt.legend()
    # %%
    alpha = zfit.Parameter("alpha", 0, -3, 3)
    bkg_yield = zfit.Parameter("bkg_yield", 15_000)
    # %%
    bkg_pdf = zfit.pdf.SplineMorphingPDF(alpha, bkg_histpdfs, extended=bkg_yield)
    # %%
    with alpha.set_value(-0.6):  # we can change this value to play around
        mplhep.histplot(bkg_pdf.to_hist())
    # %%
    # bkg_pdf = zfit.pdf.HistogramPDF(bkg_hist, extended=bkg_yield)  # we don't use the spline for simplicity
    # %%
    model = zfit.pdf.BinnedSumPDF([sig_pdf, bkg_pdf])
    model.to_hist()
    # %%
    with zfit.param.set_values(
        [alpha]
        # + list(modifiers.values())
        ,
        [0.0]
        # + list(np.random.normal(1.0, scale=0.08, size=len(modifiers)))
    ):
        # data = bkg_pdf.sample(n=12_000).to_hist() + sig_pdf.sample(3000).to_hist()
        data = model.sample()
    uncertainties = 1 / znp.maximum(template_hist.counts() ** 0.5, 1)
    print(uncertainties)
    # %%
    modifier_constraints = zfit.constraint.GaussianConstraint(
        params=list(modifiers.values()),
        observation=np.ones(len(modifiers)),
        uncertainty=uncertainties,
    )
    alpha_constraint = zfit.constraint.GaussianConstraint(alpha, 0, 1)
    # %%
    loss_binned = zfit.loss.ExtendedBinnedNLL(
        model, data, constraints=[modifier_constraints, alpha_constraint]
    )
    # %%
    result = minimizer.minimize(loss_binned)
    result.hesse()
    print(result)
    for p in loss_binned.get_params():
        print(p, p.value())
        result.errors(method="zfit_errors", params=p)
    print(result)

    # %%
    print(result)
    # %%
    plt.figure()
    mplhep.histplot(model.to_hist(), label="model")
    mplhep.histplot(data, label="data")
    plt.legend()
    # %%
    print(sig_pdf.get_yield())
    unbinned_spline = zfit.pdf.SplinePDF(sig_pdf)
    # %%
    plt.figure()
    plt.plot(x, unbinned_spline.pdf(x))
    mplhep.histplot(sig_pdf.to_hist(), density=True)
    plt.show()


test_moprhing_sum()

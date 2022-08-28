#  Copyright (c) 2022 zfit
import pytest


@pytest.mark.parametrize("n", [300, 10000, 200_000], ids=lambda x: f"samplesize={x:,}")
@pytest.mark.parametrize(
    "floatall", [True, False], ids=["floatAll", "floatSigBkgLambdaOnly"]
)
@pytest.mark.parametrize(
    "use_sampler", [True, False], ids=["useSampler", "onetimeSample"]
)
@pytest.mark.flaky(reruns=1)  # in case a fit fails
def test_sig_bkg_fit(n, floatall, use_sampler):
    #  Copyright (c) 2022 zfit

    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np

    import zfit

    n_bins = 100

    # create space
    obs_binned = zfit.Space(
        "x", binning=zfit.binned.RegularBinning(n_bins, -10, 10, name="x")
    )
    obs = obs_binned.with_binning(None)

    # parameters
    mu = zfit.Parameter("mu", 1.0, -4, 6, floating=floatall)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10, floating=floatall)

    lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)

    # model building, pdf creation
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    exponential = zfit.pdf.Exponential(lambd, obs=obs)

    n_bkg_init = 20000
    n_sig_init = 2000
    n_bkg = zfit.Parameter("n_bkg", n_bkg_init)
    n_sig = zfit.Parameter("n_sig", n_sig_init)
    gauss_extended = gauss.create_extended(n_sig)
    exp_extended = exponential.create_extended(n_bkg)
    model_unbinned = zfit.pdf.SumPDF([gauss_extended, exp_extended])

    # make binned
    model = zfit.pdf.BinnedFromUnbinnedPDF(model_unbinned, space=obs_binned)

    # data
    n_sample = n
    if use_sampler:
        data = model.create_sampler(n=n_sample)
        data.resample(n=n_sample)
        data_unbinned = model_unbinned.create_sampler(n=n_sample)
        data_unbinned.resample(n=n_sample)
    else:
        data = model.sample(n=n_sample)
        data_unbinned = model_unbinned.sample(n=n_sample)

    plot_scaling = n_sample / n_bins * obs.area()

    x = np.linspace(-10, 10, 1000)

    def plot_pdf(title):
        plt.figure()
        plt.title(title)
        y = model.pdf(x).numpy()
        y_gauss = (gauss.pdf(x) * model_unbinned.params["frac_0"]).numpy()
        y_exp = (exponential.pdf(x) * model_unbinned.params["frac_1"]).numpy()
        plt.plot(x, y * plot_scaling, label="Sum - Model")
        plt.plot(x, y_gauss * plot_scaling, label="Gauss - Signal")
        plt.plot(x, y_exp * plot_scaling, label="Exp - Background")
        mplhep.histplot(data.to_hist(), yerr=True, color="black", histtype="errorbar")
        plt.ylabel("Counts")
        plt.xlabel("obs: $B_{mass}$")
        plt.legend()

    # create NLL
    nll = zfit.loss.ExtendedBinnedNLL(model=model, data=data)
    nll_unbinned = zfit.loss.ExtendedUnbinnedNLL(
        model=model_unbinned, data=data_unbinned
    )

    # create a minimizer
    minimizer = zfit.minimize.Minuit()
    results = []
    rel = 0.03 * n**0.5 / 17  # 17 is 300 ** 0.5
    if floatall:
        rel *= 3  # higher tolerance if we float all
    for loss in [nll, nll_unbinned]:
        if floatall:
            mu.set_value(0.5)
            sigma.set_value(1.2)
        lambd.set_value(-0.05)
        if not use_sampler:
            plot_pdf(
                f"before fit - {loss.name} n:{n_sample:,} {'floatAll' if floatall else 'fixedSigShape'}"
            )

        result = minimizer.minimize(nll)
        results.append(result)
        # do the error calculations, here with hesse, than with minos
        _ = result.hesse()
        (
            param_errors,
            _,
        ) = result.errors()  # this returns a new FitResult if a new minimum was found
        # print(result.valid)  # check if the result is still valid
        # print(result)
        # plot the data
        if not use_sampler:
            plot_pdf(
                f"after fit - {loss.name} n:{n_sample:,} {'floatAll' if floatall else 'fixedSigShape'}"
            )

    assert (
        pytest.approx(results[0].params["lambda"]["value"], rel=rel)
        == results[1].params["lambda"]["value"]
    )
    assert (
        pytest.approx(results[0].params["lambda"]["hesse"], rel=rel)
        == results[1].params["lambda"]["hesse"]
    )
    assert (
        pytest.approx(results[0].params["lambda"]["errors"]["lower"], rel=rel)
        == results[1].params["lambda"]["errors"]["lower"]
    )
    assert (
        pytest.approx(results[0].params["lambda"]["errors"]["upper"], rel=rel)
        == results[1].params["lambda"]["errors"]["upper"]
    )
    assert (
        pytest.approx(results[0].params["n_bkg"]["value"], rel=rel)
        == results[1].params["n_bkg"]["value"]
    )
    assert (
        pytest.approx(results[0].params["n_bkg"]["hesse"], rel=rel)
        == results[1].params["n_bkg"]["hesse"]
    )
    assert (
        pytest.approx(results[0].params["n_bkg"]["errors"]["lower"], rel=rel)
        == results[1].params["n_bkg"]["errors"]["lower"]
    )
    assert (
        pytest.approx(results[0].params["n_bkg"]["errors"]["upper"], rel=rel)
        == results[1].params["n_bkg"]["errors"]["upper"]
    )
    assert (
        pytest.approx(results[0].params["n_sig"]["value"], rel=rel)
        == results[1].params["n_sig"]["value"]
    )
    assert (
        pytest.approx(results[0].params["n_sig"]["hesse"], rel=rel)
        == results[1].params["n_sig"]["hesse"]
    )
    assert (
        pytest.approx(results[0].params["n_sig"]["errors"]["lower"], rel=rel)
        == results[1].params["n_sig"]["errors"]["lower"]
    )
    assert (
        pytest.approx(results[0].params["n_sig"]["errors"]["upper"], rel=rel)
        == results[1].params["n_sig"]["errors"]["upper"]
    )

    if floatall:
        assert (
            pytest.approx(results[0].params["mu"]["value"], rel=rel)
            == results[1].params["mu"]["value"]
        )
        assert (
            pytest.approx(results[0].params["mu"]["hesse"], rel=rel)
            == results[1].params["mu"]["hesse"]
        )
        assert (
            pytest.approx(results[0].params["mu"]["errors"]["lower"], rel=rel)
            == results[1].params["mu"]["errors"]["lower"]
        )
        assert (
            pytest.approx(results[0].params["mu"]["errors"]["upper"], rel=rel)
            == results[1].params["mu"]["errors"]["upper"]
        )
        assert (
            pytest.approx(results[0].params["sigma"]["value"], rel=rel)
            == results[1].params["sigma"]["value"]
        )
        assert (
            pytest.approx(results[0].params["sigma"]["hesse"], rel=rel)
            == results[1].params["sigma"]["hesse"]
        )
        assert (
            pytest.approx(results[0].params["sigma"]["errors"]["lower"], rel=rel)
            == results[1].params["sigma"]["errors"]["lower"]
        )
        assert (
            pytest.approx(results[0].params["sigma"]["errors"]["upper"], rel=rel)
            == results[1].params["sigma"]["errors"]["upper"]
        )

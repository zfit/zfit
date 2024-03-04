#  Copyright (c) 2023 zfit
import pytest


@pytest.mark.parametrize("n", [500, 10000, 200_000], ids=lambda x: f"samplesize={x:,}")
@pytest.mark.parametrize(
    "floatall", [True, False], ids=["floatAll", "floatSigBkgLambdaOnly"]
)
@pytest.mark.parametrize(
    "use_sampler", [True, False], ids=["useSampler", "onetimeSample"]
)
@pytest.mark.parametrize("nbins", [27, 90, 311], ids=lambda x: f"nbins={x:,}")
@pytest.mark.parametrize(
    "use_wrapper",
    [True, False],
    ids=lambda x: f"useWrapper={x}",
)
@pytest.mark.flaky(reruns=1)  # in case a fit fails
def test_sig_bkg_fit(n, floatall, use_sampler, nbins, use_wrapper, request):
    import mplhep
    import matplotlib.pyplot as plt
    import numpy as np

    import zfit

    longtest = request.config.getoption("--longtests")
    if not longtest:
        if n > 1000 or floatall or use_sampler:
            pytest.skip("skipping long test")

    if nbins < 90 or n < 1000:
        zfit.settings.set_seed(42)

    # create space
    obs_binned = zfit.Space(
        "x", binning=zfit.binned.RegularBinning(nbins, -10, 10, name="x")
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
    if use_wrapper:
        model = zfit.pdf.BinnedFromUnbinnedPDF(model_unbinned, space=obs_binned)
    else:
        model = model_unbinned.to_binned(obs_binned)

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

    plot_scaling = n_sample / nbins * obs.area()

    x = np.linspace(-10, 10, 1000)

    plot_folder = (
        f'NLL_profile/nbins{nbins} n{n} floatall_{floatall}{" sampler" if use_sampler else ""}'
        f'{" wrapper" if use_wrapper else ""}'
    ).replace(" ", "_")

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
        pytest.zfit_savefig(folder=plot_folder)

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
    rel *= 300 / nbins
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
        if not use_sampler:
            plot_pdf(
                f"after fit - {loss.name} n:{n_sample:,} {'floatAll' if floatall else 'fixedSigShape'}"
            )

    mu.floating = True
    minimizer.minimize(nll_unbinned)
    mu.floating = False
    param_vals = np.linspace(mu.value() - 0.3, mu.value() + 0.3)
    nlls = []
    nlls_binned = []
    for val in param_vals:
        mu.set_value(val)
        try:
            minimizer.minimize(nll_unbinned)
        except Exception:
            continue

        nlls.append(nll_unbinned.value())
        try:
            minimizer.minimize(nll)
        except Exception:
            nlls.pop(-1)
            continue
        nlls_binned.append(nll.value())

    plt.figure()
    nlls = np.array(nlls) - np.min(nlls)
    nlls_binned = np.array(nlls_binned) - np.min(nlls_binned)
    plt.title(
        f'NLL profile: nbins:{nbins} n:{n} floatall:{floatall} {"sampler" if use_sampler else ""} {"wrapper" if use_wrapper else ""}'
    )
    plt.plot(param_vals, nlls, label="Unbinned")
    plt.plot(param_vals, nlls_binned, label="Binned")
    plt.legend()
    pytest.zfit_savefig(folder=plot_folder)

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


def test_nbins(request):
    #  Copyright (c) 2022 zfit

    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np
    import zfit.z.numpy as znp

    import zfit

    longtest = request.config.getoption("--longtests")

    # create space
    obs = zfit.Space("x", limits=(0, 10))

    # parameters
    init_vals = [3.1, 1.0, -0.06]
    mu = zfit.Parameter("mu", init_vals[0], -4, 10, floating=False)
    sigma = zfit.Parameter("sigma", init_vals[1], 0.1, 10, floating=False)
    lambd = zfit.Parameter("lambda", init_vals[2], -1, -0.01)

    # model building, pdf creation
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    exponential = zfit.pdf.Exponential(lambd, obs=obs)

    n_bkg = zfit.Parameter("n_bkg", 300)
    n_sig = zfit.Parameter("n_sig", 20)
    gauss_extended = gauss.create_extended(n_sig)
    exp_extended = exponential.create_extended(n_bkg)
    model_unbinned = zfit.pdf.SumPDF([gauss_extended, exp_extended])
    sig_data = model_unbinned.sample(275)
    bkg_data = model_unbinned.sample(25)
    data = zfit.Data.from_tensor(
        obs, tensor=znp.concatenate([sig_data.value(), bkg_data.value()], axis=0)
    )
    # make binned
    plot_folder = "nbins_accuracy"
    binnings = [5, 50, 1000]
    if longtest:
        binnings += [7, 10, 15, 20, 30, 102, 203, 341, 500]
        binnings = sorted(binnings)
    minimizer = zfit.minimize.Minuit()
    n_bkg_vals = []
    n_bkg_vals_unbinned = []

    for nbins in binnings:
        n_sig.floating = True
        obs_binned = obs.with_binning(nbins)
        model_binned = zfit.pdf.BinnedFromUnbinnedPDF(model_unbinned, space=obs_binned)
        data_binned = data.to_binned(obs_binned)
        # fit
        loss = zfit.loss.ExtendedUnbinnedNLL(model_unbinned, data)
        result_unbinned = minimizer.minimize(loss)
        n_bkg_vals_unbinned.append(result_unbinned.params["n_bkg"]["value"])
        loss_binned = zfit.loss.ExtendedBinnedNLL(model_binned, data_binned)
        result = minimizer.minimize(loss_binned)
        n_bkg_vals.append(result.params["n_bkg"]["value"])
        # plot
        plt.figure()
        plt.title(f"{nbins} bins full fit")
        mplhep.histplot(data_binned, alpha=0.5, yerr=False, label="data")
        mplhep.histplot(model_binned.to_hist(), label="model")
        plt.legend()
        pytest.zfit_savefig(folder=plot_folder)

        plt.figure()
        plt.title(f"{nbins} bins binned vs unbinned curve")
        x = np.linspace(0, 10, nbins)
        scaled_density = model_binned.to_hist().density() * model_binned.get_yield()
        plt.plot(x, scaled_density, "x", label="binned")
        plt.plot(x, model_unbinned.ext_pdf(x), label="unbinned")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()
        pytest.zfit_savefig(folder=plot_folder)
        # plt.show()

        if not result.valid:
            continue

    plt.figure()
    plt.title("n_bkg vs nbins")
    plt.semilogx(binnings, n_bkg_vals, "x", label="binned")
    mean = np.mean(n_bkg_vals_unbinned)
    std = np.std(n_bkg_vals_unbinned)
    plt.semilogx([np.min(binnings), np.max(binnings)], "b", [mean] * 2, label="true")
    plt.semilogx(
        [np.min(binnings), np.max(binnings)], "r", [mean] * 2, label="unbinned"
    )
    plt.semilogx(
        [np.min(binnings), np.max(binnings)],
        [mean + std] * 2,
        "r",
        alpha=0.5,
        label="unbinned + std",
    )
    plt.semilogx(
        [np.min(binnings), np.max(binnings)],
        [mean - std] * 2,
        "r",
        alpha=0.5,
        label="unbinned - std",
    )
    plt.xlabel("$N_{bins}$")
    plt.ylabel("$N_{bkg}$")
    plt.legend()
    pytest.zfit_savefig(folder=plot_folder)
    # plt.show()
    # 4 sigma away, factor of two because binned is less precise
    assert pytest.approx(mean, abs=std * 4 * 2) == result.params["n_bkg"]["value"]

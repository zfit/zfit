import pytest
import numpy as np
import time

import zfit
from zfit.minimizers.errors import WeightCorr


def create_gaussian_mixture_data(n_samples=1000, weights=True):
    """Create data for a Gaussian mixture model similar to scratch_103.py."""
    # Define the observable space
    obs = zfit.Space("x", limits=(0, 10))

    # Create parameters for the model
    mean = 5.0
    sigma1 = 0.5
    sigma2 = 1.0
    sig1frac = 0.8
    nsig = 500

    # Create signal PDFs
    sig1 = zfit.pdf.Gauss(mu=mean, sigma=sigma1, obs=obs)
    sig2 = zfit.pdf.Gauss(mu=mean, sigma=sigma2, obs=obs)

    # Combine signal components
    model = zfit.pdf.SumPDF([sig1, sig2], fracs=[sig1frac], extended=nsig)


    # Generate data
    sampler = model.create_sampler(n=n_samples)
    sampler.resample()
    data_np = sampler.value()

    if weights:
        weights_np = np.random.uniform(low=0.1, high=3.5, size=n_samples)
        data = zfit.Data.from_numpy(obs=obs, array=data_np, weights=weights_np)
    else:
        data = zfit.Data.from_numpy(obs=obs, array=data_np)

    return data, obs


def create_three_component_data(n_samples=1000, weights=True):
    """Create data for a model with three extended PDFs: Gauss, GeneralizedCB, and Exponential."""
    # Define the observable space
    obs = zfit.Space("x", limits=(0, 10))

    # Create parameters for the model
    mean_gauss = 5.0
    mean_cb = 5.2
    sigma = 0.7
    alphal = 1.0
    alphar = 1.2
    nl = 2.0
    nr = 2.5
    lam = -0.3

    # Create the PDFs
    gauss = zfit.pdf.Gauss(mu=mean_gauss, sigma=sigma, obs=obs)
    cb = zfit.pdf.GeneralizedCB(mu=mean_cb, sigmal=sigma, alphal=alphal, nl=nl,
                               sigmar=sigma, alphar=alphar, nr=nr, obs=obs)
    exp = zfit.pdf.Exponential(lam=lam, obs=obs)

    # Create extended PDFs
    n_gauss = 300
    n_cb = 400
    n_exp = 300
    gauss_ext = gauss.create_extended(n_gauss)
    cb_ext = cb.create_extended(n_cb)
    exp_ext = exp.create_extended(n_exp)

    # Create sum of extended PDFs
    model = zfit.pdf.SumPDF([gauss_ext, cb_ext, exp_ext])

    # Generate data
    sampler = model.create_sampler(n=n_samples)
    sampler.resample()
    data_np = sampler.value()

    if weights:
        # Create weights
        weights_np = np.random.uniform(low=0.1, high=3.5, size=n_samples)
        data = zfit.Data.from_numpy(obs=obs, array=data_np, weights=weights_np)
    else:
        data = zfit.Data.from_numpy(obs=obs, array=data_np)

    return data, obs


def perform_fit(data, obs, model_type="gaussian_mixture"):
    """Perform a fit with the specified model type."""
    if model_type == "gaussian_mixture":
        # Create parameters for the model
        mean = zfit.Parameter("mean", 5.3, 0.1, 10.0)
        sigma1 = zfit.Parameter("sigma1", 0.3, 0.1, 2.0)
        sigma2 = zfit.Parameter("sigma2", 1.1, 0.1, 3.0)
        sig1frac = zfit.Parameter("sig1frac", 0.9, 0.0, 1.0)
        nsig = zfit.Parameter("nsig", 500, 0, 100000)

        # Create signal PDFs
        sig1 = zfit.pdf.Gauss(mu=mean, sigma=sigma1, obs=obs)
        sig2 = zfit.pdf.Gauss(mu=mean, sigma=sigma2, obs=obs)

        # Combine signal components
        sig = zfit.pdf.SumPDF([sig1, sig2], fracs=[sig1frac])

        # Create extended signal model
        model = sig.create_extended(nsig)

    elif model_type == "three_component":
        # Create parameters for the model
        mean_gauss = zfit.Parameter("mean_gauss", 4, 0.1, 10.0)
        mean_cb = zfit.Parameter("mean_cb", 6.2, 0.1, 10.0)
        sigma = zfit.Parameter("sigma", 0.3, 0.1, 2.0)
        alphal = zfit.Parameter("alphal", 1.2, 0.1, 5.0)
        alphar = zfit.Parameter("alphar", 1.5, 0.1, 5.0)
        nl = zfit.Parameter("nl", 2.5, 0.5, 10.0)
        nr = zfit.Parameter("nr", 3.0, 0.5, 10.0)
        lam = zfit.Parameter("lambda", -0.4, -2.0, -0.1)

        n_gauss = zfit.Parameter("n_gauss", 300, 0, 100000)
        n_cb = zfit.Parameter("n_cb", 400, 0, 100000)
        n_exp = zfit.Parameter("n_exp", 300, 0, 100000)

        # Create the PDFs
        gauss = zfit.pdf.Gauss(mu=mean_gauss, sigma=sigma, obs=obs)
        cb = zfit.pdf.GeneralizedCB(mu=mean_cb, sigmal=sigma, alphal=alphal, nl=nl,
                                   sigmar=sigma, alphar=alphar, nr=nr, obs=obs)
        exp = zfit.pdf.Exponential(lam=lam, obs=obs)

        # Create extended PDFs
        gauss_ext = gauss.create_extended(n_gauss)
        cb_ext = cb.create_extended(n_cb)
        exp_ext = exp.create_extended(n_exp)

        # Create sum of extended PDFs
        model = zfit.pdf.SumPDF([gauss_ext, cb_ext, exp_ext])

    # Create loss and perform the fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit(gradient="zfit")
    result = minimizer.minimize(nll)

    return result, model








def test_compare_error_methods():
    """Test and compare different error methods for weighted fits."""
    # Create data
    data, obs = create_gaussian_mixture_data(n_samples=2000, weights=True)

    # Perform fit
    result, model = perform_fit(data, obs, model_type="gaussian_mixture")

    # Calculate parameter errors with different weight correction methods
    errors_no_corr = result.hesse(weightcorr=False, name="hesse_no_corr")
    errors_sumw2 = result.hesse(weightcorr="sumw2", name="hesse_sumw2")
    errors_asymptotic = result.hesse(weightcorr="asymptotic", name="hesse_asymptotic")

    # Check that asymptotic errors are larger than no correction
    for param_name in errors_no_corr:
        no_corr_error = errors_no_corr[param_name]["error"]
        asymptotic_error = errors_asymptotic[param_name]["error"]

        # Asymptotic errors should generally be larger than uncorrected errors for weighted fits
        assert asymptotic_error >= no_corr_error * 0.9, f"Asymptotic error for {param_name} should be larger than uncorrected"

        # Effective size errors should be between no correction and asymptotic
        sumw2_error = errors_sumw2[param_name]["error"]
        assert no_corr_error * 0.9 <= sumw2_error <= asymptotic_error * 1.1, \
            f"Effective size error for {param_name} should be between uncorrected and asymptotic"


def test_weights_one_equals_no_weights():
    """Test that all corrections give the same result if weights are all one compared to no weights given."""
    # Create data with weights=1
    n_samples = 2000
    data_with_weights, obs = create_gaussian_mixture_data(n_samples=n_samples, weights=False)
    # Create weights array of all ones
    weights_ones = np.ones(n_samples)
    data_with_weights = data_with_weights.with_weights(weights_ones)

    # Create data without weights
    data_no_weights, _ = create_gaussian_mixture_data(n_samples=n_samples, weights=False)

    # Perform fits
    result_with_weights, _ = perform_fit(data_with_weights, obs, model_type="gaussian_mixture")
    result_no_weights, _ = perform_fit(data_no_weights, obs, model_type="gaussian_mixture")

    # Calculate parameter errors with different weight correction methods
    errors_no_weights = result_no_weights.hesse(name="hesse_no_weights")
    errors_weights_no_corr = result_with_weights.hesse(weightcorr=False, name="hesse_weights_no_corr")
    errors_weights_sumw2 = result_with_weights.hesse(weightcorr="sumw2", name="hesse_weights_sumw2")
    errors_weights_asymptotic = result_with_weights.hesse(weightcorr="asymptotic", name="hesse_weights_asymptotic")

    # All error methods should give similar results when weights are all ones
    # Create a mapping of parameter names to their errors
    no_weights_errors = {param.name: info["error"] for param, info in errors_no_weights.items()}
    weights_no_corr_errors = {param.name: info["error"] for param, info in errors_weights_no_corr.items()}
    weights_sumw2_errors = {param.name: info["error"] for param, info in errors_weights_sumw2.items()}
    weights_asymptotic_errors = {param.name: info["error"] for param, info in errors_weights_asymptotic.items()}

    # Compare errors by parameter name
    for param_name in no_weights_errors:
        no_weights_error = no_weights_errors[param_name]
        weights_no_corr_error = weights_no_corr_errors[param_name]
        weights_sumw2_error = weights_sumw2_errors[param_name]
        weights_asymptotic_error = weights_asymptotic_errors[param_name]

        # All errors should be reasonably close to each other (within 100%)
        assert pytest.approx(no_weights_error, rel=1.0) == weights_no_corr_error, \
            f"No weights error and weights=1 no correction error differ for {param_name}"
        assert pytest.approx(no_weights_error, rel=1.0) == weights_sumw2_error, \
            f"No weights error and weights=1 sumw2 error differ for {param_name}"
        assert pytest.approx(no_weights_error, rel=1.0) == weights_asymptotic_error, \
            f"No weights error and weights=1 asymptotic error differ for {param_name}"


def compare_roofit_zfit_gaussian_mixture():
    """Create and fit a Gaussian mixture model with both RooFit and zfit, then compare the results."""
    # Define the observable space
    obs_z = zfit.Space("x", limits=(0, 10))

    # Create parameters for the zfit model
    mean_z = zfit.Parameter("mean_z", 5.0, 0.1, 10.0)
    sigma1_z = zfit.Parameter("sigma1_z", 0.5, 0.1, 2.0)
    sigma2_z = zfit.Parameter("sigma2_z", 1.0, 0.1, 3.0)
    sig1frac_z = zfit.Parameter("sig1frac_z", 0.8, 0.0, 1.0)
    nsig_z = zfit.Parameter("nsig_z", 500, 0, 10000)

    # Create signal PDFs for zfit
    sig1_z = zfit.pdf.Gauss(mu=mean_z, sigma=sigma1_z, obs=obs_z)
    sig2_z = zfit.pdf.Gauss(mu=mean_z, sigma=sigma2_z, obs=obs_z)

    # Combine signal components for zfit
    sig_z = zfit.pdf.SumPDF([sig1_z, sig2_z], fracs=[sig1frac_z])

    # Create extended signal model for zfit
    model_z = sig_z.create_extended(nsig_z)

    # Create RooFit observable
    x = ROOT.RooRealVar("x", "x", 0, 10)

    # Create RooFit parameters
    mean_r = ROOT.RooRealVar("mean", "mean of gaussians", 5.0, 0.1, 10.0)
    sigma1_r = ROOT.RooRealVar("sigma1", "width of gaussian 1", 0.5, 0.1, 2.0)
    sigma2_r = ROOT.RooRealVar("sigma2", "width of gaussian 2", 1.0, 0.1, 3.0)
    sig1frac_r = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
    nsig_r = ROOT.RooRealVar("nsig", "number of signal events", 500, 0, 10000)

    # Create RooFit PDFs
    sig1_r = ROOT.RooGaussian("sig1", "Signal component 1", x, mean_r, sigma1_r)
    sig2_r = ROOT.RooGaussian("sig2", "Signal component 2", x, mean_r, sigma2_r)

    # Combine signal components for RooFit
    sig_r = ROOT.RooAddPdf("sig", "Signal", [sig1_r, sig2_r], [sig1frac_r])

    # Create extended signal model for RooFit
    model_r = ROOT.RooExtendPdf("model", "extended model", sig_r, nsig_r)

    # Generate data
    n_samples = 2000
    sampler = model_z.create_sampler(n=n_samples)
    sampler.resample()
    data_np = sampler.value()

    # Create weights
    weights_np = np.random.normal(loc=1.0, scale=0.2, size=n_samples)

    # Create zfit data
    data_zfit = zfit.Data.from_numpy(obs=obs_z, array=data_np, weights=weights_np)

    # Create RooFit data
    weight_r = ROOT.RooRealVar("weight", "weight", 0.1, 1000.0)
    data_roofit = ROOT.RooDataSet("data", "data", [x], RooFit.WeightVar(weight_r))

    for val, w in zip(data_np, weights_np):
        x.setVal(val)
        weight_r.setVal(w)
        data_roofit.add([x], weight_r.getVal())

    # Reset parameters to initial values
    mean_z.set_value(5.3)
    sigma1_z.set_value(0.3)
    sigma2_z.set_value(1.1)
    sig1frac_z.set_value(0.9)

    mean_r.setVal(5.3)
    sigma1_r.setVal(0.3)
    sigma2_r.setVal(1.1)
    sig1frac_r.setVal(0.9)

    # Fit with zfit
    nll_z = zfit.loss.ExtendedUnbinnedNLL(model=model_z, data=data_zfit)
    minimizer_z = zfit.minimize.Minuit(gradient="zfit")
    result_z = minimizer_z.minimize(nll_z)
    param_errors_z = result_z.hesse(weightcorr="asymptotic", name="hesse_zfit_compare")

    # Fit with RooFit
    result_r = model_r.fitTo(data_roofit, RooFit.Save(True), RooFit.AsymptoticError(True),
                            RooFit.EvalBackend("cpu"), RooFit.Extended(True))

    # Return the results for comparison
    return {
        "zfit": {
            "nsig": (nsig_z.value(), param_errors_z[nsig_z]['error']),
            "mean": (mean_z.value(), param_errors_z[mean_z]['error']),
            "sigma1": (sigma1_z.value(), param_errors_z[sigma1_z]['error']),
            "sigma2": (sigma2_z.value(), param_errors_z[sigma2_z]['error']),
            "sig1frac": (sig1frac_z.value(), param_errors_z[sig1frac_z]['error'])
        },
        "roofit": {
            "nsig": (nsig_r.getVal(), nsig_r.getError()),
            "mean": (mean_r.getVal(), mean_r.getError()),
            "sigma1": (sigma1_r.getVal(), sigma1_r.getError()),
            "sigma2": (sigma2_r.getVal(), sigma2_r.getError()),
            "sig1frac": (sig1frac_r.getVal(), sig1frac_r.getError())
        }
    }


def test_compare_roofit_zfit_errors():
    """Test that zfit and RooFit errors are similar for the same model and data."""
    # Skip test if ROOT is not available
    ROOT = pytest.importorskip("ROOT", reason="ROOT not available")
    Roofit = ROOT.RooFit

    # Compare RooFit and zfit results
    results = compare_roofit_zfit_gaussian_mixture()

    # Check that errors are similar between RooFit and zfit
    for param_name in results["zfit"]:
        zfit_val, zfit_err = results["zfit"][param_name]
        roofit_val, roofit_err = results["roofit"][param_name]

        # Values should be similar (within 10%)
        assert pytest.approx(zfit_val, rel=0.01) == roofit_val, \
            f"zfit and RooFit values differ for {param_name}: {zfit_val} vs {roofit_val}"

        # Errors should be similar (within 20%)
        assert pytest.approx(zfit_err, rel=0.01) == roofit_err, \
            f"zfit and RooFit errors differ for {param_name}: {zfit_err} vs {roofit_err}"



def compare_roofit_zfit_three_component(weightcorr):
    """Create and fit a three-component model with both RooFit and zfit, then compare the results."""
    # Define the observable space
    ROOT = pytest.importorskip("ROOT", reason="ROOT not available")
    Roofit = ROOT.RooFit
    obs_z = zfit.Space("x", limits=(0, 10))

    # Create parameters for the zfit model
    mean_gauss_z = zfit.Parameter("mean_gauss_z", 3, 0.1, 10.0)
    mean_cb_z = zfit.Parameter("mean_cb_z", 6.2, 0.1, 10.0)
    sigma_z = zfit.Parameter("sigma_z", 0.3, 0.1, 2.0)
    alpha_z = zfit.Parameter("alpha_z", 1.0, 0.1, 5.0)
    n_z = zfit.Parameter("n_z", 2.0, 0.1, 30.0)
    lam_z = zfit.Parameter("lam_z", -0.3, -2.0, -0.1)
    n_gauss_z = zfit.Parameter("n_gauss_z", 3000, 0, 100000)
    n_cb_z = zfit.Parameter("n_cb_z", 4000, 0, 100000)
    n_exp_z = zfit.Parameter("n_exp_z", 3000, 0, 100000)

    # Create the PDFs for zfit
    gauss_z = zfit.pdf.Gauss(mu=mean_gauss_z, sigma=sigma_z, obs=obs_z)
    cb_z = zfit.pdf.CrystalBall(mu=mean_cb_z, sigma=sigma_z, alpha=alpha_z, n=n_z, obs=obs_z)
    exp_z = zfit.pdf.Exponential(lam=lam_z, obs=obs_z)

    # Create extended PDFs for zfit
    gauss_ext_z = gauss_z.create_extended(n_gauss_z)
    cb_ext_z = cb_z.create_extended(n_cb_z)
    exp_ext_z = exp_z.create_extended(n_exp_z)

    # Create sum of extended PDFs for zfit
    model_z = zfit.pdf.SumPDF([gauss_ext_z, cb_ext_z, exp_ext_z])

    # Create RooFit observable
    x = ROOT.RooRealVar("x", "x", 0, 10)

    # Create RooFit parameters - use the same initial values as zfit
    mean_gauss_r = ROOT.RooRealVar("mean_gauss", "mean of gaussian", 3.0, 0.1, 10.0)
    mean_cb_r = ROOT.RooRealVar("mean_cb", "mean of crystal ball", 6.2, 0.1, 10.0)
    sigma_r = ROOT.RooRealVar("sigma", "width of pdfs", 0.7, 0.1, 2.0)
    alpha_r = ROOT.RooRealVar("alpha", "alpha of crystal ball", 1.0, 0.1, 5.0)
    n_r = ROOT.RooRealVar("n", "n of crystal ball", 2.0, 0.2, 30.0)
    lam_r = ROOT.RooRealVar("lambda", "lambda of exponential", -0.3, -2.0, -0.1)
    n_gauss_r = ROOT.RooRealVar("n_gauss", "number of gaussian events", 3000, 0, 100000)
    n_cb_r = ROOT.RooRealVar("n_cb", "number of crystal ball events", 4000, 0, 100000)
    n_exp_r = ROOT.RooRealVar("n_exp", "number of exponential events", 3000, 0, 100000)

    # Create RooFit PDFs
    gauss_r = ROOT.RooGaussian("gauss", "Gaussian component", x, mean_gauss_r, sigma_r)
    cb_r = ROOT.RooCBShape("cb", "Crystal Ball component", x, mean_cb_r, sigma_r, alpha_r, n_r)
    exp_r = ROOT.RooExponential("exp", "Exponential component", x, lam_r)

    # Create extended PDFs for RooFit
    gauss_ext_r = ROOT.RooExtendPdf("gauss_ext", "extended gaussian", gauss_r, n_gauss_r)
    cb_ext_r = ROOT.RooExtendPdf("cb_ext", "extended crystal ball", cb_r, n_cb_r)
    exp_ext_r = ROOT.RooExtendPdf("exp_ext", "extended exponential", exp_r, n_exp_r)

    # Create sum of extended PDFs for RooFit
    model_r = ROOT.RooAddPdf("model", "three component model", [gauss_ext_r, cb_ext_r, exp_ext_r])

    # Generate data
    n_samples = 9_000
    sample = model_z.sample(n=n_samples)
    data_np = sample.value()[:, 0]

    # Create weights
    weights_np = np.random.uniform(0.2, 3.5, size=n_samples)

    # Create zfit data
    data_zfit = zfit.Data.from_numpy(obs=obs_z, array=data_np, weights=weights_np)

    # Create RooFit data
    weight_r = ROOT.RooRealVar("weight", "weight", 0.1, 1000.0)
    data_roofit = ROOT.RooDataSet("data", "data", [x], RooFit.WeightVar(weight_r))

    for val, w in zip(data_np, weights_np):
        x.setVal(val)
        weight_r.setVal(w)
        data_roofit.add([x], weight_r.getVal())

    # Reset parameters to initial values
    mean_gauss_z.set_value(4.7)
    mean_cb_z.set_value(7.5)
    sigma_z.set_value(0.8)
    alpha_z.set_value(1.2)
    n_z.set_value(2.5)
    lam_z.set_value(-0.4)

    mean_gauss_r.setVal(4.7)
    mean_cb_r.setVal(7.5)
    sigma_r.setVal(0.8)
    alpha_r.setVal(1.2)
    n_r.setVal(2.5)
    lam_r.setVal(-0.4)

    # Fit with zfit
    nll_z = zfit.loss.ExtendedUnbinnedNLL(model=model_z, data=data_zfit)
    minimizer_z = zfit.minimize.Minuit()
    result_z = minimizer_z.minimize(nll_z)
    param_errors_z = result_z.hesse(weightcorr=weightcorr, name="hesse_zfit_three_compare")

    # # Plot the fitted PDF
    # x_plot = np.linspace(0, 10, 1000)
    # y_plot = model_z.pdf(x_plot)
    # # Get component PDFs
    # fracs = list(model_z.params.values())
    # y_gauss = gauss_ext_z.pdf(x_plot) * fracs[0]
    # y_cb = cb_ext_z.pdf(x_plot) * fracs[1]
    # y_exp = exp_ext_z.pdf(x_plot) * fracs[2]
    #
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(10,6))
    # plt.hist(data_np, bins=50, weights=weights_np, density=True, alpha=0.5, label='Data')
    # plt.plot(x_plot, y_plot, 'k-', label='Total Fit', linewidth=2)
    # plt.plot(x_plot, y_gauss, 'r--', label='Gaussian', linewidth=2)
    # plt.plot(x_plot, y_cb, 'g--', label='Crystal Ball', linewidth=2)
    # plt.plot(x_plot, y_exp, 'b--', label='Exponential', linewidth=2)
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('Density')
    # plt.title('Three Component Fit')
    # plt.show()

    # Fit with RooFit
    if weightcorr:
        # Set the weight correction method
        if weightcorr == "sumw2":
            weightcorr_r = ROOT.RooFit.SumW2Error(True)
        elif weightcorr == "asymptotic":
            weightcorr_r = ROOT.RooFit.AsymptoticError(True)
        else:
            raise ValueError(f"Unknown weight correction method: {weightcorr}")
    else:
        weightcorr_r = ROOT.RooFit.SumW2Error(False)
    result_r = model_r.fitTo(data_roofit, RooFit.Save(True), weightcorr_r,
                            RooFit.EvalBackend("cpu"), RooFit.Extended(True))


    # Print comparison of results
    # print("\nComparison of zfit and RooFit results:")
    # print("-" * 80)
    # print(f"{'Parameter':<15} {'zfit value':>12} {'zfit error':>12} {'RooFit value':>12} {'RooFit error':>12}")
    # print("-" * 80)

    # Return the results for comparison
    resultvals = {
        "zfit": {
            "n_gauss": (n_gauss_z.value(), param_errors_z[n_gauss_z]['error']),
            "n_cb": (n_cb_z.value(), param_errors_z[n_cb_z]['error']),
            "n_exp": (n_exp_z.value(), param_errors_z[n_exp_z]['error']),
            "mean_gauss": (mean_gauss_z.value(), param_errors_z[mean_gauss_z]['error']),
            "mean_cb": (mean_cb_z.value(), param_errors_z[mean_cb_z]['error']),
            "sigma": (sigma_z.value(), param_errors_z[sigma_z]['error']),
            "alpha": (alpha_z.value(), param_errors_z[alpha_z]['error']),
            "n": (n_z.value(), param_errors_z[n_z]['error']),
            "lambda": (lam_z.value(), param_errors_z[lam_z]['error'])
        },
        "roofit": {
            "n_gauss": (n_gauss_r.getVal(), n_gauss_r.getError()),
            "n_cb": (n_cb_r.getVal(), n_cb_r.getError()),
            "n_exp": (n_exp_r.getVal(), n_exp_r.getError()),
            "mean_gauss": (mean_gauss_r.getVal(), mean_gauss_r.getError()),
            "mean_cb": (mean_cb_r.getVal(), mean_cb_r.getError()),
            "sigma": (sigma_r.getVal(), sigma_r.getError()),
            "alpha": (alpha_r.getVal(), alpha_r.getError()),
            "n": (n_r.getVal(), n_r.getError()),
            "lambda": (lam_r.getVal(), lam_r.getError())
        }
    }

    # # Print the comparison
    # for param in resultvals["zfit"]:
    #     zfit_val, zfit_err = resultvals["zfit"][param]
    #     roofit_val, roofit_err = resultvals["roofit"][param]
    #     print(f"{param:<15} {zfit_val:>12.4f} {zfit_err:>12.4f} {roofit_val:>12.4f} {roofit_err:>12.4f}")
    # print("-" * 80)
    #
    # # Plot comparison of values and errors
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # param_names = list(resultvals["zfit"].keys())
    # xvals = np.arange(len(param_names))
    # width = 0.35
    #
    # # Plot parameter values with errors
    # for i, param in enumerate(param_names):
    #     # Create twin axis for each parameter
    #     ax = plt.subplot(len(param_names), 1, i+1)
    #
    #     # Plot points with error bars
    #     zfit_val, zfit_err = resultvals["zfit"][param]
    #     roofit_val, roofit_err = resultvals["roofit"][param]
    #
    #     ax.errorbar([0], [zfit_val], yerr=[zfit_err], fmt='o', label='zfit', markersize=8)
    #     ax.errorbar([0.2], [roofit_val], yerr=[roofit_err], fmt='s', label='RooFit', markersize=8)
    #
    #     # Customize axis
    #     ax.set_xlim(-0.5, 0.7)
    #     ax.set_xticks([])
    #     ax.set_ylabel(param)
    #
    #     # Add legend only to first subplot
    #     if i == 0:
    #         ax.legend()
    #
    # plt.tight_layout()
    # plt.show()
    # # Create RooFit frame and plot
    # frame = x.frame(ROOT.RooFit.Title("Three Component Fit - RooFit"))
    # data_roofit.plotOn(frame)
    # model_r.plotOn(frame)
    # model_r.plotOn(frame, ROOT.RooFit.Components("gauss_ext"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
    # model_r.plotOn(frame, ROOT.RooFit.Components("cb_ext"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen))
    # model_r.plotOn(frame, ROOT.RooFit.Components("exp_ext"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))
    #
    # # Create canvas and draw
    # c = ROOT.TCanvas("c", "c", 800, 600)
    # frame.Draw()
    #
    # # Add legend
    # legend = ROOT.TLegend(0.65, 0.75, 0.89, 0.89)
    # legend.AddEntry(frame.findObject("model_Norm[x]"), "Total Fit", "l")
    # legend.AddEntry(frame.findObject("gauss_ext_Norm[x]"), "Gaussian", "l")
    # legend.AddEntry(frame.findObject("cb_ext_Norm[x]"), "Crystal Ball", "l")
    # legend.AddEntry(frame.findObject("exp_ext_Norm[x]"), "Exponential", "l")
    # legend.Draw()
    #
    # c.Update()
    # c.Draw()
    #

    return resultvals

@pytest.mark.parametrize("weightcorr", [False, "sumw2", "asymptotic"])
def test_compare_roofit_zfit_three_component_errors(weightcorr):
    """Test that zfit and RooFit errors are similar for the three-component model."""

    ROOT = pytest.importorskip("ROOT", reason="ROOT not available")
    Roofit = ROOT.RooFit
    # Compare RooFit and zfit results
    results = compare_roofit_zfit_three_component(weightcorr=weightcorr)

    # Check that values and errors are similar between RooFit and zfit
    for param_name in results["zfit"]:
        zfit_val, zfit_err = results["zfit"][param_name]
        roofit_val, roofit_err = results["roofit"][param_name]

        # For all parameters, compare values with a tight tolerance (3%)
        assert pytest.approx(zfit_val, rel=0.03) == roofit_val, \
            f"zfit and RooFit values differ for {param_name}: {zfit_val} vs {roofit_val}"

        relerr = 0.15 if weightcorr == "sumw2" else 0.03  # only approximate, it's not correct.
        # we don't do the squared weights in the NLL calculation, just multiply the weights with the pdf vals
        # and then multiply by the sum of weights and divide by the sum of squares.
        assert pytest.approx(zfit_err, rel=relerr) == roofit_err, \
            f"zfit and RooFit errors differ significantly for {param_name}: {zfit_err} vs {roofit_err}"

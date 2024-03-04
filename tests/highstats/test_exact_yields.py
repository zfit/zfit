#  Copyright (c) 2023 zfit
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize("exact_nsample", [True, False], ids=["exact", "binomial sum"])
def test_yield_bias(exact_nsample, ntoys=300):
    import zfit
    import zfit.z.numpy as znp

    plot_folder = pathlib.Path(
        f"{'exact' if exact_nsample else 'binomial_sum'}_yield_sampling"
    )

    obs = zfit.Space("x", limits=(-10, 10))
    mu = zfit.Parameter("mu", 1.0, -4, 6)
    sigma = zfit.Parameter("sigma", 1.1, 0.1, 10)
    lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    exponential = zfit.pdf.Exponential(lambd, obs=obs)
    n_bkg = zfit.Parameter("n_bkg", 20000)
    n_sig = zfit.Parameter("n_sig", 1000)
    params = [mu, sigma, lambd, n_bkg, n_sig]
    gauss_extended = gauss.create_extended(n_sig)
    exp_extended = exponential.create_extended(n_bkg)
    model = zfit.pdf.SumPDF([gauss_extended, exp_extended])
    init_vals = [0.5, 1.2, -0.05, 20350, 2512]
    true_vals = [1.0, 1.1, -0.06, 20000, 3000]
    zfit.param.set_values(params, true_vals)
    true_nsig = true_vals[-1]
    gauss_sample = gauss.create_sampler(n=true_nsig)
    true_nbkg = true_vals[-2]
    exp_sample = exponential.create_sampler(n=true_nbkg)

    def sample_func():
        exp_sample.resample()
        gauss_sample.resample()
        gauss_val = gauss_sample.value()
        assert gauss_val.shape[0] == true_nsig
        exp_val = exp_sample.value()
        assert exp_val.shape[0] == true_nbkg
        return znp.concatenate([gauss_val, exp_val])

    data = model.create_sampler()
    data.sample_holder.assign(sample_func())
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    nsigs = []
    nbkgs = []
    minimizer = zfit.minimize.Minuit(gradient=False, tol=1e-05, mode=2)
    failures = 0
    for _ in range(ntoys):
        zfit.param.set_values(params, true_vals)
        if exact_nsample:
            data.sample_holder.assign(sample_func())
        else:
            data.resample(n=sum(true_vals[-2:]))
        for _ in range(10):
            zfit.param.set_values(params, init_vals)
            rnd_sig = np.random.uniform(-(true_nsig**0.5), true_nsig**0.5)
            rnd_bkg = np.random.uniform(-(true_nbkg**0.5), true_nbkg**0.5)
            rnd_sig *= (rnd_sig > 0) + 1
            rnd_sig += (rnd_sig > 0) - 0.5 + true_nsig**0.5 / 2
            rnd_bkg *= (rnd_bkg > 0) + 1
            rnd_bkg += (rnd_bkg > 0) - 0.5 + true_nbkg**0.5 / 2
            n_sig.set_value(true_nsig + rnd_sig)
            n_bkg.set_value(true_nbkg + rnd_bkg)
            result = minimizer.minimize(nll)
            if result.valid:
                break
        else:
            failures += 1
            continue
        nsig_res = float(result.params[n_sig]["value"])
        nsigs.append(nsig_res)
        nbkg_res = float(result.params[n_bkg]["value"])
        nbkgs.append(nbkg_res)
        assert nsig_res + nbkg_res == pytest.approx(true_nsig + true_nbkg, abs=0.6)
    nsigs_mean = np.mean(nsigs)
    std_nsigs_mean = np.std(nsigs) / ntoys**0.5 * 5
    nbkg_mean = np.mean(nbkgs)
    std_nbkg_mean = np.std(nbkgs) / ntoys**0.5 * 5
    plt.figure("yield_bias_toys")
    plt.title(
        f'{"Exact" if exact_nsample else "Binomial sum"} sampled. Fit with {minimizer.name}.'
    )

    counts, edges, _ = plt.hist(nsigs, bins=50, label=" Signal yields", alpha=0.5)
    npoints = 50
    plt.plot(
        np.ones(npoints) * true_nsig, np.linspace(0, np.max(counts)), "gx", label="true"
    )

    plt.plot(
        np.ones(npoints) * nsigs_mean,
        np.linspace(0, np.max(counts) * 2 / 3),
        "bo",
        label=f"Signal mean: {nsigs_mean:.2f} +- {std_nsigs_mean:.2f}",
    )

    plt.plot(
        np.ones(npoints) * nsigs_mean - np.std(nsigs),
        np.linspace(0, np.max(counts) * 0.2),
        "ro",
        label="-std",
    )

    plt.plot(
        np.ones(npoints) * nsigs_mean + np.std(nsigs),
        np.linspace(0, np.max(counts) * 0.2),
        "ro",
        label="+std",
    )

    plt.legend()
    pytest.zfit_savefig(folder=plot_folder)
    rel_err_sig = 0.001
    assert nsigs_mean == pytest.approx(true_nsig, rel=rel_err_sig, abs=std_nsigs_mean)
    rel_err_bkg = 0.001
    assert nbkg_mean == pytest.approx(true_nbkg, rel=rel_err_bkg, abs=std_nbkg_mean)

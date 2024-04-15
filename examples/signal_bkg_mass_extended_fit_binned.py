#  Copyright (c) 2024 zfit
from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep
import numpy as np

import zfit

plt.style.use(mplhep.style.LHCb2)
n_bins = 50
# create space
obs_binned = zfit.Space("x", binning=zfit.binned.RegularBinning(50, -10, 10, name="x"))
obs = obs_binned.with_binning(None)  # unbinned obs

# parameters, model building, pdf creation
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
nr = zfit.Parameter("nr", 2, 0, 5)
alpha = zfit.Parameter("alpha", 1.5, 0, 5)
n_sig = zfit.Parameter("n_sig", 5000 * 0.3)
doublecb = zfit.pdf.GeneralizedCB(
    mu=mu,
    sigmal=sigma,
    nl=nr,
    alphal=alpha,  # sharing sigma, meaning the width is the same left and right
    sigmar=sigma,
    nr=nr,
    alphar=0.5,
    obs=obs,
    extended=n_sig,
)

lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
n_bkg = zfit.Parameter("n_bkg", 5000 * (1 - 0.3))
exponential = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)

# since `obs_binned` is binned, giving it to the pdf will return a binned pdf
# note that this is different from binning both pdfs and summing them (there, we would need a BinnedSumPDF)
model = zfit.pdf.SumPDF([doublecb, exponential], obs=obs_binned)

# data
n_sample = 5000
data = model.sample(n=n_sample)

plot_scaling = n_sample / n_bins * obs.volume

x = np.linspace(obs.v1.lower, obs.v1.upper, 1000)


def plot_pdf(title):
    plt.figure()
    plt.title(title)
    y = model.pdf(x)
    y_doublecb = doublecb.pdf(x) * model.pdfs[0].params["frac_0"]
    y_exp = exponential.pdf(x) * model.pdfs[0].params["frac_1"]
    plt.plot(x, y * plot_scaling, label="Sum - Binned Model")
    plt.plot(x, y_doublecb * plot_scaling, label="Gauss - Signal")
    plt.plot(x, y_exp * plot_scaling, label="Exp - Background")
    # mplhep.histplot(np.histogram(data_np, bins=n_bins), yerr=True, color='black', histtype='errorbar')
    mplhep.histplot(data, yerr=data.counts() ** 0.5, color="black", histtype="errorbar")
    plt.ylabel("Counts")
    plt.xlabel("obs: $B_{mass}$")
    plt.legend()
    # plt.savefig(title + ".pdf")  # uncomment to save the plot


# set the values to a start value for the fit
zfit.param.set_values({mu: 0.5, sigma: 1.2, nr: 2.0, alpha: 1.5, lambd: -0.05})
# alternatively, we can set the values with a list/array
# zfit.param.set_values([mu, sigma, nr, alpha, lambd], [0.5, 1.2, 2.0, 1.5, -0.05])

# create NLL
nll = zfit.loss.ExtendedBinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit(gradient="zfit")

plot_pdf("before fit")

result = minimizer.minimize(nll).update_params()
# do the error calculations, here with hesse, than with minos

param_hesse = result.hesse()
(
    param_errors,
    _,
) = result.errors()  # this returns a new FitResult if a new minimum was found

# this returns a new FitResult if a new minimum was found
param_errors, _ = result.errors()
# plot the data and pdf
plot_pdf("after fit")
# uncomment to display plots
# plt.show()

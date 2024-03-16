#  Copyright (c) 2024 zfit

import matplotlib.pyplot as plt
import mplhep
import numpy as np

import zfit

plt.style.use(mplhep.style.LHCb2)

n_bins = 50

# create space
obs_binned = zfit.Space("x", binning=zfit.binned.RegularBinning(50, -10, 10, name="x"))
obs = obs_binned.with_binning(None)

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
nr = zfit.Parameter("nr", 2, 0, 5)
alpha = zfit.Parameter("alpha", 1.5, 0, 5)
gauss = zfit.pdf.DoubleCB(mu=mu, sigma=sigma, nl=nr, nr=nr, alphal=alpha, alphar=0.5, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)

n_bkg = zfit.Parameter("n_bkg", 5000 * (1 - 0.3))
n_sig = zfit.Parameter("n_sig", 5000 * 0.3)
gauss_extended = gauss.create_extended(n_sig)
exp_extended = exponential.create_extended(n_bkg)
model_unbinned = zfit.pdf.SumPDF([gauss_extended, exp_extended])

# make binned
model = zfit.pdf.BinnedFromUnbinnedPDF(model_unbinned, space=obs_binned)

# data
n_sample = 5000
data = model.sample(n=n_sample)

n_bins = 50

plot_scaling = n_sample / n_bins * obs.area()

x = np.linspace(-10, 10, 1000)


def plot_pdf(title):
    plt.figure()
    # plt.title(title)
    y = model.pdf(x).numpy()
    y_gauss = (gauss.pdf(x) * model_unbinned.params["frac_0"]).numpy()
    y_exp = (exponential.pdf(x) * model_unbinned.params["frac_1"]).numpy()
    plt.plot(x, y * plot_scaling, label="Sum - Binned Model")
    plt.plot(x, y_gauss * plot_scaling, label="Gauss - Signal")
    plt.plot(x, y_exp * plot_scaling, label="Exp - Background")
    # mplhep.histplot(np.histogram(data_np, bins=n_bins), yerr=True, color='black', histtype='errorbar')
    h = data.to_hist()
    mplhep.histplot(h, yerr=h.counts() ** 0.5, color="black", histtype="errorbar")
    plt.ylabel("Counts")
    plt.xlabel("obs: $B_{mass}$")
    plt.legend()
    plt.savefig(title + ".pdf")


# set the values to a start value for the fit
mu.set_value(0.5)
sigma.set_value(1.2)
lambd.set_value(-0.05)

# create NLL
nll = zfit.loss.ExtendedBinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()

plot_pdf("before fit")

result = minimizer.minimize(nll)
# do the error calculations, here with hesse, than with minos
param_hesse = result.hesse()
(
    param_errors,
    _,
) = result.errors()  # this returns a new FitResult if a new minimum was found

# plot the data

plot_pdf("after fit")
# uncomment to display plots
plt.show()

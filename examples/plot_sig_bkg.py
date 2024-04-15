#  Copyright (c) 2024 zfit
from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep
import numpy as np

import zfit

mplhep.style.use("LHCb2")
# create space
obs = zfit.Space("x", -10, 10)

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
frac = zfit.Parameter("fraction", 0.3, 0, 1)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)
model = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

# for a quick plot, we can use the attached "plotter"
plt.figure()
plt.title("Model before fit, using plotter")
model.plot.plotpdf()

# data
n_sample = 5000

exp_data = exponential.sample(n=n_sample * (1 - frac))

gauss_data = gauss.sample(n=n_sample * frac)

data = model.create_sampler(n_sample, limits=obs)
data.resample()

# set the values of the parameters
zfit.param.set_values([mu, sigma, lambd, frac], [0.5, 1.2, -0.05, 0.07])

# alternatively, we can set the values individually
# mu.set_value(0.5)
# sigma.set_value(1.2)
# lambd.set_value(-0.05)
# frac.set_value(0.07)

# plot the data
n_bins = 50

plot_scaling = n_sample / n_bins * obs.volume

x = np.linspace(obs.v1.lower, obs.v1.upper, 1000)


def plot_pdf(title):
    plt.figure()
    plt.title(title)
    y = model.pdf(x)
    y_gauss = gauss.pdf(x) * frac
    y_exp = exponential.pdf(x) * (1 - frac)
    plt.plot(x, y * plot_scaling, label="Sum - Model")
    plt.plot(x, y_gauss * plot_scaling, label="Gauss - Signal")
    plt.plot(x, y_exp * plot_scaling, label="Exp - Background")
    mplhep.histplot(
        data.to_binned(n_bins),
        yerr=True,
        color="black",
        histtype="errorbar",
    )
    plt.ylabel("Counts")
    plt.xlabel("$B_{mass}$")


# plot the pdf BEFORE fitting
plot_pdf("Before fitting")


# create NLL
nll = zfit.loss.UnbinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll).update_params()

# do the error calculations, here with minos
param_errors, _ = result.errors()

plot_pdf(title="After fitting")
plt.legend()

# uncomment to display plots
# plt.show()

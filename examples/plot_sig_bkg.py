#  Copyright (c) 2023 zfit

import mplhep
import numpy as np

import zfit

mplhep.style.use("LHCb2")
import matplotlib.pyplot as plt

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
frac = zfit.Parameter("fraction", 0.3, 0, 1)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)
model = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

# data
n_sample = 10000

exp_data = exponential.sample(n=n_sample * (1 - frac)).numpy()

gauss_data = gauss.sample(n=n_sample * frac).numpy()

data = model.create_sampler(n_sample, limits=obs)
data.resample()

mu.set_value(0.5)
sigma.set_value(1.2)
lambd.set_value(-0.05)
frac.set_value(0.07)

# plot the data
data_np = data["x"].numpy()
n_bins = 50

plot_scaling = n_sample / n_bins * obs.area()

x = np.linspace(-10, 10, 1000)


def plot_pdf(title):
    plt.figure()
    plt.title(title)
    y = model.pdf(x).numpy()
    y_gauss = (gauss.pdf(x) * frac).numpy()
    y_exp = (exponential.pdf(x) * (1 - frac)).numpy()
    plt.plot(x, y * plot_scaling, label="Sum - Model")
    plt.plot(x, y_gauss * plot_scaling, label="Gauss - Signal")
    plt.plot(x, y_exp * plot_scaling, label="Exp - Background")
    mplhep.histplot(
        np.histogram(data_np, bins=n_bins),
        yerr=True,
        color="black",
        histtype="errorbar",
    )
    plt.ylabel("Counts")
    plt.xlabel("obs: $B_{mass}$")


# plot the pdf BEFORE fitting
plot_pdf("Before fitting")
# create NLL
nll = zfit.loss.UnbinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

# do the error calculations, here with minos
param_errors, _ = result.errors()

plot_pdf(title="After fitting")
plt.legend()

# uncomment to display plots
# plt.show()

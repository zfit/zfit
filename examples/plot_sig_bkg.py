#  Copyright (c) 2019 zfit

import numpy as np
import zfit
import matplotlib.pyplot as plt

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1., -4, 6)
sigma = zfit.Parameter("sigma", 1., 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
frac = zfit.Parameter("fraction", 0.3, 0, 1)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)
model = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

# data
n_sample = 10000

exp_data = zfit.run(exponential.sample(n=n_sample * (1 - frac)))

gauss_data = zfit.run(gauss.sample(n=n_sample * frac))

data = model.create_sampler(n_sample, limits=obs)
data.resample()

mu.set_value(0.5)
sigma.set_value(1.2)
lambd.set_value(-0.05)
frac.set_value(0.07)

# plot the data
data_np = zfit.run(data[:, 0])
color = 'black'
n_bins = 50

linewidth = 2.5
plot_scaling = n_sample / n_bins * obs.area()

x = np.linspace(-10, 10, 1000)

# plot the pdf BEFORE fitting
plt.figure()
plt.title("Before fitting")
# plot the data
plt.hist(data_np, color=color, bins=n_bins, histtype="stepfilled", alpha=0.1)
plt.hist(data_np, color=color, bins=n_bins, histtype="step")
# plot the pdfs
y = zfit.run(model.pdf(x))
y_gauss = zfit.run(gauss.pdf(x) * frac)  # notice the frac!
y_exp = zfit.run(exponential.pdf(x) * (1 - frac))  # notice the frac!

plt.plot(x, y * plot_scaling, label="Sum - Model", linewidth=linewidth * 2)
plt.plot(x, y_gauss * plot_scaling, '--', label="Gauss - Signal", linewidth=linewidth)
plt.plot(x, y_exp * plot_scaling, '--', label="Exponential - Background", linewidth=linewidth)
plt.xlabel("Physical observable")
plt.legend()

# create NLL
nll = zfit.loss.UnbinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

# do the error calculations, here with minos
param_errors = result.error()

plt.figure()
plt.title("After fitting")
# plot the data
plt.hist(data_np, color=color, bins=n_bins, histtype="stepfilled", alpha=0.1)
plt.hist(data_np, color=color, bins=n_bins, histtype="step")

y = zfit.run(model.pdf(x))  # rerun now after the fitting
y_gauss = zfit.run(gauss.pdf(x) * frac)
y_exp = zfit.run(exponential.pdf(x) * (1 - frac))

plt.plot(x, y * plot_scaling, label="Sum - Model", linewidth=linewidth * 2)
plt.plot(x, y_gauss * plot_scaling, '--', label="Gauss - Signal", linewidth=linewidth)
plt.plot(x, y_exp * plot_scaling, '--', label="Exponential - Background", linewidth=linewidth)
plt.xlabel("Physical observable")
plt.legend()

plt.show()

#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import zfit

zfit.run.experimental_disable_param_update(True)
# Create space
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

raw_data = model.sample(n=10_000, params={mu: 1.2, frac: 0.4, sigma: 1.5, lambd: -0.05})
# Create weights directly
raw_values = raw_data.value()[:, 0]
weights = np.random.uniform(0.5, 3.5, size=raw_values.shape[0])

# Create weighted dataset directly by providing weights
weighted_data = zfit.Data(data=raw_values, obs=obs, weights=weights)

# Create a weighted loss - now automatically handles the weighted data
nll = zfit.loss.UnbinnedNLL(model=model, data=weighted_data)

# Minimize with proper handling of weights
minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll).update_params()  # updates the default value

# plot the pdf

weighted_data.to_binned(50).to_hist().plot(density=True)
model.plot.plotpdf()
plt.show()

# Error estimation with weighted data correction
# The default is "asymptotic" correction, but we can use other methods
param_errors_none = result.hesse(name="nocorr", weightcorr=False)  # Uses effective size method
param_errors_effsize = result.hesse(name="effsize_corr", weightcorr="effsize")  # Uses effective size method
param_errors_asymp = result.hesse(name="asymp_corr", weightcorr="asymptotic")  # Asymptotic correction
param_errors_aysmetric = result.errors()

print(result)

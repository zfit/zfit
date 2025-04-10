#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np

import zfit

zfit.run.experimental_disable_param_update(True)

# Create space and parameters
obs = zfit.Space("x", -10, 10)
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
n_sig = zfit.Parameter("n_sig", 300, 0, 1000)
n_bkg = zfit.Parameter("n_bkg", 700, 0, 2000)

# Create extended PDFs
signal = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=n_sig)
background = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)
model = zfit.pdf.SumPDF([signal, background])

# Define explicit parameter values for sampling
sampling_params = {
    mu: 1.2,  # Initial: 1.0
    sigma: 1.1,  # Initial: 1.0
    lambd: -0.08,  # Initial: -0.06
    n_sig: 350,  # Initial: 300
    n_bkg: 650,  # Initial: 700
}

# Sample with explicit parameters
data = model.sample(n=1000, params=sampling_params)

# Generate weights based on position
x_values = data.value()
weights = 1.0 + 0.2 * np.sin(x_values)  # Position-dependent weights

# Create weighted dataset
weighted_data = data.with_weights(weights)

# Create NLL with weighted data
nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=weighted_data)

# Minimize with weight-aware gradient
minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll).update_params()
result.hesse()

print(result)

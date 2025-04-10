#  Copyright (c) 2024 zfit
from __future__ import annotations

import zfit

zfit.run.experimental_disable_param_update(True)

# Create space
obs = zfit.Space("x", -10, 10)

# Define parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)

# Create a PDF
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# Explicitly set different parameter values for sampling
# Instead of using the current parameter values
sampling_params = {
    mu: 1.3,  # Slightly different from mu's initial value of 1.0
    sigma: 1.2,  # Slightly different from sigma's initial value of 1.0
}

# Sample with explicit parameter values
data = gauss.sample(n=1000, params=sampling_params)


# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit(gradient=False)
result = minimizer.minimize(nll).update_params()

# do the error calculations with a hessian approximation
param_errors = result.hesse()

# or here with minos
param_errors_asymmetric, new_result = result.errors()

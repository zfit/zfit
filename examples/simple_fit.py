#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np

import zfit

# create space
obs = zfit.Space("x", -2, 3)

# parameters
mu = zfit.Parameter("mu", 1.2, -4, 6)
sigma = zfit.Parameter("sigma", 1.3, 0.5, 10)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# data
data = np.random.normal(loc=2.0, scale=3.0, size=10000)
# we can convert it to a zfit Data object to make sure the data is within the limits or use the space to filter manually
# data = obs.filter(data)  # works also for pandas DataFrame
data = zfit.Data(obs=obs, data=data)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit(gradient=False)
result = minimizer.minimize(nll).update_params()

# do the error calculations with a hessian approximation
param_errors = result.hesse()

# or here with minos
param_errors_asymmetric, new_result = result.errors()

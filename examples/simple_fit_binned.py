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
obs_binned = obs.with_binning(30)
gauss_binned = gauss.to_binned(obs_binned)

normal_np = np.random.normal(loc=2.0, scale=3.0, size=10000)
data = zfit.Data.from_numpy(obs=obs, array=normal_np)
data_binned = data.to_binned(obs_binned)  # could also be a hist object

nll = zfit.loss.BinnedNLL(model=gauss_binned, data=data_binned)

minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll).update_params()

param_errors = result.hesse()

# or here with minos
param_errors_asymmetric, new_result = result.errors()

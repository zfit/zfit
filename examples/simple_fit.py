#  Copyright (c) 2024 zfit
from __future__ import annotations

import time

import numpy as np

import zfit

# TODO: revert to default script! WIP
# create space
obs = zfit.Space("x", -2, 3)

# parameters
mu = zfit.Parameter("mu", 1.2, -4, 6)
sigma = zfit.Parameter("sigma", 1.3, 0.5, 10)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# data
data = np.random.normal(loc=2.0, scale=3.0, size=10_000 * 100)
# we can convert it to a zfit Data object to make sure the data is within the limits or use the space to filter manually
data = obs.filter(data)  # works also for pandas DataFrame
# data = zfit.Data.from_numpy(obs=obs, array=data)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)
nll.value()
nll.value_gradient()
nll.hessian()

# create a minimizer
tol = 1e-5
minimizer2 = zfit.minimize.Minuit(gradient=True, verbosity=7, tol=tol)
minimizer = zfit.minimize.LevenbergMarquardt(verbosity=7, tol=tol)
start = time.time()
params = nll.get_params()
with zfit.param.set_values(params, params):
    result = minimizer.minimize(nll)
print("Fit time", time.time() - start)
start = time.time()
print(result)
with zfit.param.set_values(params, params):
    result2 = minimizer2.minimize(nll)
print("Fit time", time.time() - start)
print(result2)
result.update_params()

# do the error calculations with a hessian approximation
param_errors = result.hesse()

# or here with minos
param_errors_asymmetric, new_result = result.errors()

#  Copyright (c) 2022 zfit

import numpy as np

import zfit

# create space
obs = zfit.Space("x", limits=(-2, 3))

# parameters
mu = zfit.Parameter("mu", 1.2, -4, 6)
sigma = zfit.Parameter("sigma", 1.3, 0.5, 10)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# data
normal_np = np.random.normal(loc=2.0, scale=3.0, size=10000)
data = zfit.Data.from_numpy(obs=obs, array=normal_np)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit(gradient=False)
result = minimizer.minimize(nll)

# do the error calculations with a hessian approximation
param_errors = result.hesse()

# or here with minos
param_errors_asymetric, new_result = result.errors()

print(result)

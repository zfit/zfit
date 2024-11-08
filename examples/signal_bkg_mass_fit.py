#  Copyright (c) 2024 zfit
from __future__ import annotations

import zfit

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

# data
n_sample = 10000

# if we sample once, otherwise use create_sampler
data = model.sample(n_sample, limits=obs)  # limits can be omitted, then the default limits are used

# set the values to a start value for the fit
zfit.param.set_values({mu: 0.5, sigma: 1.2, lambd: -0.05, frac: 0.07})

# create NLL
nll = zfit.loss.UnbinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
zfit.run.experimental_disable_param_update()
result = minimizer.minimize(nll).update_params()
# forward compatibility, update_params will update the values of the parameters to the minimum


# do the error calculations, here with minos
param_hesse = result.hesse()
param_errors, new_result = result.errors()

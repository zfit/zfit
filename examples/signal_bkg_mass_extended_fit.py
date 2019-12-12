#  Copyright (c) 2019 zfit

import numpy as np
import zfit
import matplotlib.pyplot as plt

n_bins = 50

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1., -4, 6)
sigma = zfit.Parameter("sigma", 1., 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)

n_bkg = zfit.Parameter('n_bkg', 20000)
n_sig = zfit.Parameter('n_sig', 1000)
gauss_extended = gauss.create_extended(n_sig)
exp_extended = exponential.create_extended(n_bkg)
model = zfit.pdf.SumPDF([gauss_extended, exp_extended])

# data
# n_sample = 10000

data = model.create_sampler(n=21200)
data.resample()

# set the values to a start value for the fit
mu.set_value(0.5)
sigma.set_value(1.2)
lambd.set_value(-0.05)

# create NLL
nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)
print(result.params)
# do the error calculations, here with minos
param_errors = result.error()

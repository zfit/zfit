#  Copyright (c) 2024 zfit
from __future__ import annotations

import pickle

import zfit

n_bins = 50

# create space
obs = zfit.Space("x", -10, 10)

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
n_bkg = zfit.Parameter("n_bkg", 20000)
n_sig = zfit.Parameter("n_sig", 1000)

# model building, pdf creation
gauss_extended = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=n_sig)
exp_extended = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)

model = zfit.pdf.SumPDF([gauss_extended, exp_extended])

data = model.create_sampler(n=21200)
data.resample()

# set the values to a start value for the fit
zfit.param.set_values({mu: 0.5, sigma: 1.2, lambd: -0.05})
# alternatively, we can set the values individually
# mu.set_value(0.5)
# sigma.set_value(1.2)
# lambd.set_value(-0.05)

# create NLL
nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)

# create a minimizer
# this uses the automatic gradient (more precise, but can fail)
minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll).update_params()
# do the error calculations, here with hesse, than with minos
param_hesse = result.hesse()

# the second return value is a new FitResult if a new minimum was found

(
    param_errors,
    _,
) = result.errors()

# Storing the result can be achieved in many ways. Using dill (like pickle), we can dump and load the result
result_dilled = zfit.dill.dumps(result)
result_loaded = zfit.dill.loads(result_dilled)

# EXPERIMENTAL: we can serialize the model to a human-readable format with HS3
# human readable representation
hs3like = zfit.hs3.dumps(nll)
# print(hs3like)
# and we can load it again
nll_loaded = zfit.hs3.loads(hs3like)

# pickle can also be used, but as not all objects are pickleable, we need to freeze the result
# which makes it read-only and no new error calculations can be done
result.freeze()
dumped = pickle.dumps(result)
loaded = pickle.loads(dumped)

zfit.param.set_values(model.get_params(), loaded)

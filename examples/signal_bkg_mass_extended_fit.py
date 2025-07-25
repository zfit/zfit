#  Copyright (c) 2025 zfit
from __future__ import annotations

import zfit

zfit.run.experimental_disable_param_update(True)

n_bins = 50

# create space
obs = zfit.Space("x", -10, 10)

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
n_bkg = zfit.Parameter("n_bkg", 20000, 0, 50000)
n_sig = zfit.Parameter("n_sig", 1000, 0, 30000)

# model building, pdf creation
gauss_extended = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=n_sig)
exp_extended = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)

model = zfit.pdf.SumPDF([gauss_extended, exp_extended])

n = 21200
data = model.sample(n=n)

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

param_errors, _ = result.errors()

print(result)
# Storing the result can be achieved in many ways. Using dill (like pickle), we can dump and load the result
result_dilled = zfit.dill.dumps(result)
result_loaded = zfit.dill.loads(result_dilled)

zfit.param.set_values(model.get_params(), result_loaded)

# EXPERIMENTAL: we can serialize the model to a human-readable format with HS3
# human readable representation
hs3like = zfit.hs3.dumps(nll)
# print(hs3like)
# and we can load it again
loaded = zfit.hs3.loads(hs3like)
print(loaded)

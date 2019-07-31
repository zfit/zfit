#  Copyright (c) 2019 zfit

import numpy as np
import zfit

# create space
obs = zfit.Space("x", limits=(-2, 3))

# parameters
mu = zfit.Parameter("mu", 1.2, -4, 6)
sigma = zfit.Parameter("sigma", 1.3, 0.1, 10)

# model building, pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# data
normal_np = np.random.normal(loc=2., scale=3., size=10000)
data = zfit.Data.from_numpy(obs=obs, array=normal_np)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

# do the error calculations, here with minos
param_errors = result.error()


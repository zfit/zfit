#  Copyright (c) 2022 zfit

# create space
import numpy as np

import zfit

obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu_shared = zfit.Parameter("mu_shared", 1.0, -4, 6)  # mu is a shared parameter
sigma1 = zfit.Parameter("sigma_one", 1.0, 0.1, 10)
sigma2 = zfit.Parameter("sigma_two", 1.0, 0.1, 10)

# model building, pdf creation
gauss1 = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma1, obs=obs)
gauss2 = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma2, obs=obs)

# data
normal_np = np.random.normal(loc=2.0, scale=3.0, size=10000)
data1 = zfit.Data.from_numpy(obs=obs, array=normal_np)

# data
normal_np = np.random.normal(loc=2.0, scale=4.0, size=10000)
data2 = zfit.Data.from_numpy(obs=obs, array=normal_np)

# create simultaenous loss, two possibilities
nll_simultaneous = zfit.loss.UnbinnedNLL(model=[gauss1, gauss2], data=[data1, data2])
# OR, equivalently
nll1 = zfit.loss.UnbinnedNLL(model=gauss1, data=data1)
nll2 = zfit.loss.UnbinnedNLL(model=gauss2, data=data2)
nll_simultaneous2 = nll1 + nll2

minimizer = zfit.minimize.Minuit()
minimizer.minimize(nll_simultaneous2)

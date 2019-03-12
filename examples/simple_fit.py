import numpy as np
import zfit

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1., -4, 6)
sigma = zfit.Parameter("sigma", 1., 0.1, 10)

# pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

# data
data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(loc=2., scale=3., size=10000))

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.MinuitMinimizer()
result = minimizer.minimize(nll)

# do the error calculations, here with minos
param_errors = result.error()

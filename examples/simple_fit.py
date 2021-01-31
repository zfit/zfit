#  Copyright (c) 2021 zfit

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
normal_np = np.random.normal(loc=2., scale=3., size=10000)
data = zfit.Data.from_numpy(obs=obs, array=normal_np)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

params = [mu, sigma]


def func(**kwargs):
    zfit.param.set_values(params, kwargs.values())
    val = nll.value()
    if not np.isfinite(val):
        val = 999999999
    return - val


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {p.name: (p.lower, p.upper) for p in params}

optimizer = BayesianOptimization(
    f=func,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=20,
    n_iter=300,
)

print(optimizer.max)
# print(optimizer.res)

# # create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)
print(result)
print(result.fmin)

# do the error calculations, here with minos
# param_errors, _ = result.errors()

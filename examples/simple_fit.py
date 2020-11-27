#  Copyright (c) 2021 zfit
from collections import OrderedDict

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

# # create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

# do the error calculations with a hessian approximation
param_errors = result.hesse()

# or here with minos
param_errors_asymetric, _ = result.errors()

import dill
import cloudpickle as clp

dmps = {}


def dump_recurse(obj):
    obj_dict = obj.__dict__ if not isinstance(obj, dict) else obj
    for k, val in obj_dict.items():
        if k == '_cachers':
            val = 'Nothing'
        print(k, val)
        if isinstance(val, (dict, OrderedDict)):
            dmp1 = dump_recurse(val)
        else:
            dmp1 = clp.dumps(val)
            # dmp1 = dill.dumps(val, recurse=True)
        dmps[k] = dmp1
    return dmps


dump_recurse(gauss)
dump_recurse(data)
dump_recurse(nll)
# dump_recurse(minimizer)

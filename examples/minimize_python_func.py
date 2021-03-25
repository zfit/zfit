#  Copyright (c) 2021 zfit

import numpy as np

import zfit

# set everything to numpy mode
zfit.run.set_autograd_mode(False)
zfit.run.set_graph_mode(False)

# create our favorite minimizer
minimizer = zfit.minimize.IpyoptV1()


# we can also use a more complicated function
# from scipy.optimize import rosen as func

def func(x):
    x = np.array(x)  # make sure it's an array
    return np.sum((x - 0.1) ** 2 + x[1] ** 4)


# we need to set the errordef, the definition of "1 sigma"
func.errordef = 0.5

# initial parameters
params = [1, -3, 2, 1.4, 11]

# minimize
result = minimizer.minimize(func, params)

print(result)
# estimate errors
result.hesse()
result.errors()
print(result)

#  Copyright (c) 2022 zfit

import numpy as np

import zfit

# create space
xobs = zfit.Space("xobs", (-4, 4))
yobs = zfit.Space("yobs", (-3, 5))
zobs = zfit.Space("z", (-2, 4))
obs = xobs * yobs * zobs

# parameters
mu1 = zfit.Parameter("mu1", 1.0, -4, 6)
mu23 = zfit.Parameter("mu_shared", 1.0, -4, 6)
sigma12 = zfit.Parameter("sigma_shared", 1.0, 0.1, 10)
sigma3 = zfit.Parameter("sigma3", 1.0, 0.1, 10)

# model building, pdf creation
gauss_x = zfit.pdf.Gauss(mu=mu1, sigma=sigma12, obs=xobs)
gauss_y = zfit.pdf.Gauss(mu=mu23, sigma=sigma12, obs=yobs)
gauss_z = zfit.pdf.Gauss(mu=mu23, sigma=sigma3, obs=zobs)

product_gauss = zfit.pdf.ProductPDF([gauss_x, gauss_y, gauss_z])

# OR create directly from your 3 dimensional pdf as
# model = MyPDF(obs=obs, param1=..., param2,...)

# data
normal_np = np.random.normal(loc=[2.0, 2.5, 2.5], scale=[3.0, 3, 1.5], size=(10000, 3))
data_raw = zfit.Data.from_numpy(
    obs=obs, array=normal_np
)  # or from anywhere else, e.g. root

df = data_raw.to_pandas()
# preprocessing here, rename things. Match column names with the observable names "xobs", "yobs", "z" (they have to be
# contained, more columns in the df is not a problem)
data = zfit.Data.from_pandas(df, obs=obs)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=product_gauss, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)
print(result.params)

# do the error calculations, here with minos
param_errors, _ = result.errors()
print(param_errors)

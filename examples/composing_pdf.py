#  Copyright (c) 2023 zfit

import zfit

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1.0, -4, 6)
sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
lambd = zfit.Parameter("lambda", -1.0, -5.0, 0)
frac = zfit.Parameter("fraction", 0.5, 0.0, 1.0)

# pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)

sum_pdf = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

free_param = zfit.Parameter("free", 5, 0, 15)
param_shift = zfit.ComposedParameter("comp", lambda x: x + 5, free_param)

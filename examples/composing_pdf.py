#  Copyright (c) 2019 zfit

import zfit

# create space
obs = zfit.Space("x", limits=(-10, 10))

# parameters
mu = zfit.Parameter("mu", 1., -4, 6)
sigma = zfit.Parameter("sigma", 1., 0.1, 10)
lambd = zfit.Parameter("lambda", -1., -5., 0)
frac = zfit.Parameter("fraction", 0.5, 0., 1.)

# pdf creation
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)

# two equivalent ways to create a sum with a fraction
sum_pdf = frac * gauss + exponential
# OR
sum_pdf = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

#  Copyright (c) 2024 zfit
from __future__ import annotations

import tensorflow as tf

import zfit

# IMPORTANT! The communication of which axis corresponds to which data point happens here. So the user knows now that
# he should create this pdf with a space in the obs (x, y, z).


class CustomPDF(zfit.pdf.ZPDF):
    """3-dimensional PDF calculating doing something fancy, takes x, y, z as data."""

    _PARAMS = ("alpha", "beta")  # specify which parameters to take
    _N_OBS = 3

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):  # implement function
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        alpha = params["alpha"]
        beta = params["beta"]
        x0_new = tf.math.cos(alpha) * x0
        x1_new = tf.math.sinh(beta) * x1
        x2_new = x2 + 4.2
        return x0_new**2 + x1_new**2 + x2_new**2


xobs = zfit.Space("xobs", -4, 4)
yobs = zfit.Space("yobs", -3, 3)
zobs = zfit.Space("z", -2, 2)
obs = xobs * yobs * zobs

alpha = zfit.Parameter("alpha", 0.2)  # floating
beta = zfit.Parameter("beta", 0.4, floating=False)  # non-floating
custom_pdf = CustomPDF(obs=obs, alpha=alpha, beta=beta)

integral = custom_pdf.integrate(limits=obs)  # = 1 since normalized
sample = custom_pdf.sample(n=1000)  # DO NOT USE THIS FOR TOYS!
prob = custom_pdf.pdf(sample)  # DO NOT USE THIS FOR TOYS!

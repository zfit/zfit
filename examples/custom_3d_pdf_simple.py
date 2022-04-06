#  Copyright (c) 2022 zfit

import tensorflow as tf

import zfit
from zfit import z


# IMPORTANT! The communication of which axis corresponds to which data point happens here. So the user knows now that
# he should create this pdf with a space in the obs (x, y, z).


class CustomPDF(zfit.pdf.ZPDF):
    """3-dimensional PDF calculating doing something fancy, takes x, y, z as data."""

    _PARAMS = ["alpha", "beta"]  # specify which parameters to take
    _N_OBS = 3

    def _unnormalized_pdf(self, x):  # implement function
        x, y, z = x.unstack_x()
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        x_new = tf.math.cos(alpha) * x
        y_new = tf.math.sinh(beta) * y
        z_new = z + 4.2
        return x_new**2 + y_new**2 + z_new**2


xobs = zfit.Space("xobs", (-4, 4))
yobs = zfit.Space("yobs", (-3, 3))
zobs = zfit.Space("z", (-2, 2))
obs = xobs * yobs * zobs

alpha = zfit.Parameter("alpha", 0.2)  # floating
beta = zfit.Parameter("beta", 0.4, floating=False)  # non-floating
custom_pdf = CustomPDF(obs=obs, alpha=alpha, beta=beta)

integral = custom_pdf.integrate(limits=obs)  # = 1 since normalized
sample = custom_pdf.sample(n=1000)  # DO NOT USE THIS FOR TOYS!
prob = custom_pdf.pdf(sample)  # DO NOT USE THIS FOR TOYS!

integral_np, sample_np, prob_np = [integral.numpy(), sample.numpy(), prob.numpy()]

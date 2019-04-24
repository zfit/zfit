#  Copyright (c) 2019 zfit

import zfit
from zfit import ztf
import tensorflow as tf


# IMPORTANT! The communication of which axis corresponds to which data point happens here. So the user knows now that
# he should create this pdf with a space in the obs (x, y, z).


class CustomPDF(zfit.pdf.ZPDF):
    """3-dimensional PDF calculating doing something fancy. Takes x, y, z as data."""
    _PARAMS = ['alpha', 'beta']  # specify which parameters to take
    _N_OBS = 3

    def _unnormalized_pdf(self, x):  # implement function
        x, y, z = x.unstack_x()
        alpha = self.params['alpha']
        beta = self.params['beta']
        x_new = tf.math.cos(alpha) * x
        y_new = tf.math.sinh(beta) * y
        z_new = z + 4.2
        return x_new ** 2 + y_new ** 2 + z_new ** 2


obs = zfit.Space(["x", "y", "z"], limits=(((-4., -3., -2.),), ((4., 3., 2.),)))

custom_pdf = CustomPDF(obs=obs, alpha=0.2, beta=0.4)

integral = custom_pdf.integrate(limits=obs)  # = 1 since normalized
sample = custom_pdf.sample(n=1000)
prob = custom_pdf.pdf(sample)  # DO NOT USE THIS FOR TOYS!

integral_np, sample_np, prob_np = zfit.run([integral, sample, prob])

#  Copyright (c) 2019 zfit

import numpy as np
import zfit
import tensorflow as tf


class ParabolaPDF(zfit.pdf.ZPDF):
    """1-dimensional PDF implementing the exp(alpha * x) shape."""
    _PARAMS = ['scale', 'shift']  # specify which parameters to take

    def _unnormalized_pdf(self, x):  # implement function
        data = x.unstack_x()
        scale = self.params['scale']
        shift = self.params['shift']
        squared = tf.square(data - shift)
        return scale * squared


# create space
obs = zfit.Space("x", limits=(-2, 3))

# parameters
shift = zfit.Parameter("shift", 1.2, -4, 6)
scale = zfit.Parameter("scale", 1.3, 0.1, 10)

# model building, pdf creation
parabola = ParabolaPDF(shift=shift, scale=scale, obs=obs)

# data
normal_np = np.random.uniform(size=2000)  # TODO: data as numpy array. Or load it with other format.
data = zfit.Data.from_numpy(obs=obs, array=normal_np)

# create NLL
nll = zfit.loss.UnbinnedNLL(model=parabola, data=data)

# create a minimizer
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

print(f"value of shift: {result.params[shift]['value']}")
print(f"value of scale: {result.params[scale]['value']}")

# do the error calculations, here with minos. If needed
param_errors = result.error()

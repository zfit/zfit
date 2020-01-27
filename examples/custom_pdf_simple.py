#  Copyright (c) 2019 zfit

import zfit
from zfit import z


class CustomPDF(zfit.pdf.ZPDF):
    """1-dimensional PDF implementing the exp(alpha * x) shape."""
    _PARAMS = ['alpha']  # specify which parameters to take

    def _unnormalized_pdf(self, x):  # implement function
        data = x.unstack_x()
        alpha = self.params['alpha']

        return z.exp(alpha * data)


obs = zfit.Space("obs1", limits=(-4, 4))

custom_pdf = CustomPDF(obs=obs, alpha=0.2)

integral = custom_pdf.integrate(limits=(-1, 2))
sample = custom_pdf.sample(n=1000)
prob = custom_pdf.pdf(sample)

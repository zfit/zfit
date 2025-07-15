#  Copyright (c) 2024 zfit
from __future__ import annotations

import zfit
import zfit.z.numpy as znp


class CustomPDF(zfit.pdf.ZPDF):
    """1-dimensional PDF implementing the exp(alpha * x) shape."""

    _PARAMS = ("alpha",)  # specify which parameters to take

    @zfit.supports(norm=False)
    def _pdf(self, x, norm, params):  # implement function
        del norm
        data = x[0]  # axis 0
        alpha = params["alpha"]

        return znp.exp(alpha * data)


obs = zfit.Space("obs1", -4, 4)

custom_pdf = CustomPDF(obs=obs, alpha=0.2)

integral = custom_pdf.integrate(limits=(-1, 2))
sample = custom_pdf.sample(n=1000)
prob = custom_pdf.pdf(sample)

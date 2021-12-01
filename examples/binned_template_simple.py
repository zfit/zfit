#  Copyright (c) 2021 zfit
import hist
import mplhep
import numpy as np
from matplotlib import pyplot as plt

import zfit

# noinspection PyTypeChecker
h = hist.Hist(hist.axis.Regular(9, -3, 2, name="x", flow=False))
x = np.random.normal(size=1_000_000)
h.fill(x=x)

ntot = zfit.Parameter("ntot", 1_000)
pdf = zfit.pdf.HistogramPDF(h, extended=ntot)

h_back = pdf.to_hist()

mplhep.histplot(h_back)
# plt.legend()

plt.show()
print(h_back)

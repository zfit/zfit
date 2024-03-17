#  Copyright (c) 2022 zfit
import hist
import mplhep
import numpy as np
from matplotlib import pyplot as plt

import zfit

# noinspection PyTypeChecker
histos = []
for i in range(5):
    h = hist.Hist(hist.axis.Regular(13, -3, 2, name="x", flow=False))
    x = np.random.normal(size=1_000_000 * (i + 1)) + i**1.5 / 2 * ((-1) ** i)
    h.fill(x=x)
    histos.append(h)
mplhep.histplot(
    histos, stack=True, histtype="fill", label=[f"process {i + 1}" for i in range(5)]
)
plt.legend()
pdfs = [zfit.pdf.HistogramPDF(h) for h in histos]
sumpdf = zfit.pdf.BinnedSumPDF(pdfs)


h_back = sumpdf.to_hist()
pdf_syst = zfit.pdf.BinwiseScaleModifier(sumpdf, modifiers=True)

mplhep.histplot(h_back)

print(h_back)
# uncomment to show plots
# plt.show()

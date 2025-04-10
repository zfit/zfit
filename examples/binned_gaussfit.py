#  Copyright (c) 2024 zfit
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import zfit

zfit.run.experimental_disable_param_update(True)

# Create space with binning
obs_binned = zfit.Space("x", -2, 3, binning=30)

# Parameters
mu = zfit.Parameter("mu", 1.2, -4, 6)
sigma = zfit.Parameter("sigma", 1.3, 0.5, 10)

# Create unbinned PDF
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_binned)

# Generate data and convert to binned
normal_np = np.random.normal(loc=2.0, scale=3.0, size=10000)
data = zfit.Data(obs=obs_binned, data=normal_np)

# Create binned NLL
nll = zfit.loss.BinnedNLL(model=gauss, data=data)

# Create binned chi-square
chi2 = zfit.loss.BinnedChi2(model=gauss, data=data)

# Minimize with gradient
minimizer = zfit.minimize.Minuit(gradient="zfit")
result = minimizer.minimize(nll)

# Calculate parameter errors
param_errors = result.hesse()
param_errors_asymmetric, new_result = result.errors()

# Compare two different losses
result_chi2 = minimizer.minimize(chi2).update_params()  # only if wanted

plt.figure()
plt.title("Comparison of NLL and $\chi^2$ fit")
data.to_hist().plot(density=True)
gauss.plot.plotpdf(label="$\chi^2$", full=False)  # don't plot labels yet
result.update_params()
gauss.plot.plotpdf(label="NLL")


print("Result of NLL fit:", result)
print("Result of Chi2 fit:", result_chi2)
plt.show()

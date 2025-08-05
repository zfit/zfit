"""Examples demonstrating Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

from pathlib import Path

import arviz as az

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
import matplotlib.pyplot as plt

import zfit

# Setup
zfit.settings.set_seed(42)

here = Path(__file__).parent
plotpath = here / "plots" / "bayesian"
plotpath.mkdir(parents=True, exist_ok=True)

# ===== 1. Model setup =====
obs = zfit.Space("mass", limits=(4.0, 6.0))

# Simple Gaussian signal + exponential background
mu = zfit.Parameter("mu", 5.0, 4.5, 5.5, prior=zfit.prior.Uniform(lower=4.8, upper=5.2))
sigma = zfit.Parameter("sigma", 0.1, 0.05, 0.3, prior=zfit.prior.HalfNormal(sigma=0.1))
lambda_bkg = zfit.Parameter("lambda_bkg", -1.0, -3.0, 0.0, prior=zfit.prior.Normal(mu=-1.0, sigma=0.5))

# Yield parameters
n_sig = zfit.Parameter("n_sig", 1000, 0, 5000, prior=zfit.prior.Normal(mu=1000, sigma=100))
n_bkg = zfit.Parameter("n_bkg", 500, 0, 2000, prior=zfit.prior.Normal(mu=500, sigma=50))

# Create PDFs
signal = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_bkg, extended=n_bkg)
model = zfit.pdf.SumPDF([signal, background])

# ===== 2. Generate synthetic data =====
# True parameters
true_params = {"mu": 5.0, "sigma": 0.12, "lambda_bkg": -1.2, "n_sig": 1000, "n_bkg": 500}

# Set true values and generate data
zfit.param.set_values([mu, sigma, lambda_bkg, n_sig, n_bkg], true_params)

n_events = int(n_sig.value() + n_bkg.value())
data = model.sample(n_events)

# Reset to different starting values
starting_values = {"mu": 4.95, "sigma": 0.15, "lambda_bkg": -0.8, "n_sig": 900, "n_bkg": 600}
zfit.param.set_values([mu, sigma, lambda_bkg, n_sig, n_bkg], starting_values)

# ===== 3. Set up Bayesian inference =====
nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
params = [mu, sigma, lambda_bkg, n_sig, n_bkg]

# Use emcee sampler
sampler = zfit.mcmc.EmceeSampler(nwalkers=16, verbosity=0)

# ===== 4. Run MCMC sampling =====
print("Running MCMC sampling...")
posterior = sampler.sample(loss=nll, params=params, n_samples=500, n_warmup=200)

# ===== 5. Display results =====
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(posterior)

# ===== 6. Key methods demonstration =====
print("\n" + "-" * 40)
print("KEY METHODS")
print("-" * 40)

# Basic statistics
print(f"Mean of mu: {posterior.mean('mu'):.4f}")
print(f"Std of mu: {posterior.std('mu'):.4f}")
print(f"95% CI for mu: {posterior.credible_interval('mu', alpha=0.05)}")

# Convergence check
print(f"\nConvergence status: {posterior.converged}")
print(f"Valid samples: {posterior.valid}")

# ArviZ integration
print("\nArviZ integration:")
idata = posterior.to_arviz()
print(f"InferenceData groups: {list(idata.groups())}")
print("\nArviZ summary (first 3 params):")
print(az.summary(idata, var_names=["mu", "sigma", "lambda_bkg"]))


# ===== 7. Simple visualization =====
print("\n" + "-" * 40)
print("VISUALIZATION")
print("-" * 40)

# Simple posterior plot for key parameters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# Plot mu posterior
samples_mu = posterior.get_samples("mu")
ax1.hist(samples_mu, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black")
ax1.axvline(true_params["mu"], color="red", linestyle="--", linewidth=2, label="True value")
ax1.axvline(posterior.mean("mu"), color="green", linestyle="-", linewidth=2, label="Posterior mean")
ax1.set_xlabel("mu")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot sigma posterior
samples_sigma = posterior.get_samples("sigma")
ax2.hist(samples_sigma, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black")
ax2.axvline(true_params["sigma"], color="red", linestyle="--", linewidth=2, label="True value")
ax2.axvline(posterior.mean("sigma"), color="green", linestyle="-", linewidth=2, label="Posterior mean")
ax2.set_xlabel("sigma")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plotpath / "posterior_distributions.png", dpi=300)
print("Saved posterior plot to:", plotpath / "posterior_distributions.png")
plt.close(fig)

# Plot model fit
plt.figure(figsize=(12, 8))

# hep.histplot(data_binned.to_binned(50), histtype="errorbar", color="black", label="Data")

# Set parameters to posterior means
# explicit
# posterior_means = {param.name: posterior.mean(param.name) for param in params}
# zfit.param.set_values(params, posterior_means)
# using the posterior
posterior.update_params()

model.plot.plotpdf(data, full=True)
# # Plot model components
# x_plot = np.linspace(4.0, 6.0, 1000)
# x_plot_data = zfit.Data.from_numpy(obs=obs, array=x_plot.reshape(-1, 1))
#
# # model.plot.plotpdf(x_plot_data, full=False)
# # Normalized PDFs
# total_yield = n_sig.value() + n_bkg.value()
# sig_frac = n_sig.value() / total_yield
# bkg_frac = n_bkg.value() / total_yield
#
# y_total = model.pdf(x_plot_data)
# y_sig = signal.pdf(x_plot_data) * sig_frac
# y_bkg = background.pdf(x_plot_data) * bkg_frac
#
# plt.plot(x_plot, y_total, "r-", linewidth=2, label="Total fit")
# plt.plot(x_plot, y_sig, "g--", linewidth=2, label="Signal component")
# plt.plot(x_plot, y_bkg, "b--", linewidth=2, label="Background component")

plt.xlabel("Mass")
plt.ylabel("Probability Density")
plt.title("Bayesian Fit Result (Posterior Means)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(plotpath / "model_fit.png", dpi=300)
print("Saved model fit plot to:", plotpath / "model_fit.png")
plt.close("all")

# ===== 8. ArviZ plotting =====
print("\n" + "-" * 40)
print("ARVIZ INTEGRATION")
print("-" * 40)

# Trace plots
az.plot_trace(idata, figsize=(14, 8))
plt.suptitle("MCMC Trace Plots", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(plotpath / "trace_plots.png", dpi=300, bbox_inches="tight")
print(f"Saved trace plot to: {plotpath / 'trace_plots.png'}")
plt.close()

# Posterior distributions with credible intervals
az.plot_posterior(idata, figsize=(12, 8), hdi_prob=0.95)
plt.suptitle("Posterior Distributions with 95% HDI", fontsize=16)
plt.tight_layout()
plt.savefig(plotpath / "posterior_plots.png", dpi=300, bbox_inches="tight")
print(f"Saved posterior distributions to: {plotpath / 'posterior_plots.png'}")
plt.close()

# Parameter correlation corner plot
az.plot_pair(idata, figsize=(12, 12), kind="scatter", marginals=True, textsize=12)
plt.suptitle("Parameter Correlations", fontsize=16)
plt.tight_layout()
plt.savefig(plotpath / "corner_plot.png", dpi=300, bbox_inches="tight")
print(f"Saved corner plot to: {plotpath / 'corner_plot.png'}")
plt.close()

# Autocorrelation diagnostics
az.plot_autocorr(idata, figsize=(14, 6))
plt.suptitle("Autocorrelation Functions", fontsize=16)
plt.tight_layout()
plt.savefig(plotpath / "autocorr_plots.png", dpi=300, bbox_inches="tight")
print(f"Saved autocorrelation plots to: {plotpath / 'autocorr_plots.png'}")
plt.close()


# Convert posterior to prior for hierarchical modeling
mu_posterior_prior = posterior.as_prior("mu")  # name

# Covariance matrix
cov_matrix = posterior.covariance()

print(f"\nParameter covariance matrix shape: {cov_matrix.shape}")

# Context manager for setting parameters to posterior mean
print("\nUsing context manager to set parameters to posterior means:")
mu.set_value(4.95)
print(f"Original mu value: {mu.value():.4f}")
with posterior:
    print(f"Inside context: mu = {mu.value():.4f} (set to posterior mean)")
print(f"After context: mu = {mu.value():.4f} (restored)")

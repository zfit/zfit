"""Examples demonstrating Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import os
import time

from zfit._mcmc import NUTSSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import zfit
import zfit.z.numpy as znp

# Import Bayesian components
from zfit.mcmc import (
    EmceeSampler,
    Posteriors,
)
from zfit.prior import HalfNormalPrior, NormalPrior, UniformPrior


# Let's create a simple Gaussian model with some data
def example_gaussian_inference():
    """Example of Bayesian inference for a Gaussian model."""

    print("\n=== Bayesian inference for a Gaussian model ===\n")

    # Create data
    true_mu = 2.0
    true_sigma = 1.5
    n_data = 500

    # Set a random seed for reproducibility
    # np.random.seed(42)

    # Generate data
    data_np = np.random.normal(true_mu, true_sigma, size=n_data)
    obs = zfit.Space("x", true_mu - 10 * true_sigma, true_mu + 10 * true_sigma)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    # Define priors
    mu_prior = NormalPrior(mu=0.0, sigma=2.0)
    sigma_prior = HalfNormalPrior(mu=0, sigma=2.0)

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=mu_prior)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=sigma_prior)

    # Create model
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create negative log-likelihood loss
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Perform maximum likelihood fit first (for comparison)
    minimizer = zfit.minimize.Minuit()
    result_ml = minimizer.minimize(nll)
    result_ml.hesse()

    print("Maximum Likelihood results:")
    print(f"mu = {result_ml.params[mu]['value']:.4f} ± {result_ml.params[mu]['hesse']['error']:.4f}")
    print(f"sigma = {result_ml.params[sigma]['value']:.4f} ± {result_ml.params[sigma]['hesse']['error']:.4f}")

    # Now perform Bayesian inference
    sampler = EmceeSampler()
    posterior = sampler.sample(nll, n_samples=5000, n_warmup=1000)

    # Print summary
    posterior.print_summary()

    # Plot the results
    plt.figure(figsize=(15, 8))

    # Plot trace for mu
    plt.subplot(2, 3, 1)
    posterior.plot_trace(mu)

    # Plot trace for sigma
    plt.subplot(2, 3, 2)
    posterior.plot_trace(sigma)

    # Plot joint posterior
    plt.subplot(2, 3, 3)
    posterior.plot_pair(mu, sigma)

    # Plot posterior for mu
    plt.subplot(2, 3, 4)
    posterior.plot_posterior(mu)

    # Plot posterior for sigma
    plt.subplot(2, 3, 5)
    posterior.plot_posterior(sigma)

    # Plot the data with model
    plt.subplot(2, 3, 6)
    x = np.linspace(-5, 10, 1000)

    # Data histogram
    plt.hist(data_np, bins=30, density=True, alpha=0.5, label="Data")

    # ML estimate
    mu_ml = result_ml.params[mu]["value"]
    sigma_ml = result_ml.params[sigma]["value"]
    y_ml = 1 / (sigma_ml * np.sqrt(2 * np.pi)) * np.exp(-((x - mu_ml) ** 2) / (2 * sigma_ml**2))
    plt.plot(x, y_ml, "r-", label=f"ML: μ={mu_ml:.2f}, σ={sigma_ml:.2f}")

    # Bayesian posterior mean
    mu_bayes = posterior.mean(mu)
    sigma_bayes = posterior.mean(sigma)
    y_bayes = 1 / (sigma_bayes * np.sqrt(2 * np.pi)) * np.exp(-((x - mu_bayes) ** 2) / (2 * sigma_bayes**2))
    plt.plot(x, y_bayes, "g--", label=f"Bayes: μ={mu_bayes:.2f}, σ={sigma_bayes:.2f}")

    # True values
    y_true = 1 / (true_sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - true_mu) ** 2) / (2 * true_sigma**2))
    plt.plot(x, y_true, "k:", label=f"True: μ={true_mu:.2f}, σ={true_sigma:.2f}")

    plt.xlabel(obs)
    plt.ylabel("Density")
    plt.title("Gaussian Model Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("gaussian_inference.png")
    plt.show()

    return posterior


def example_model_comparison():
    """Example of Bayesian model comparison."""

    print("\n=== Bayesian model comparison ===\n")

    # Create data from a mixture of two Gaussians
    # np.random.seed(42)

    # True parameters
    true_mu1 = 0.0
    true_sigma1 = 1.0
    true_mu2 = 5.0
    true_sigma2 = 1.5
    true_frac = 0.7

    n_data = 500

    # Generate the data
    component = np.random.choice(2, size=n_data, p=[true_frac, 1 - true_frac])
    data_np = np.zeros(n_data)
    mask1 = component == 0
    mask2 = component == 1
    data_np[mask1] = np.random.normal(true_mu1, true_sigma1, size=np.sum(mask1))
    data_np[mask2] = np.random.normal(true_mu2, true_sigma2, size=np.sum(mask2))

    obs = zfit.Space("x", -30, 30)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    # Model 1: Single Gaussian
    mu1 = zfit.Parameter("mu1_model1", 0.0, -10.0, 10.0, prior=NormalPrior(mu=0.0, sigma=5.0))
    sigma1 = zfit.Parameter("sigma1_model1", 2.0, 0.1, 10.0, prior=HalfNormalPrior(mu=0.0, sigma=5.0))

    gauss_model = zfit.pdf.Gauss(mu=mu1, sigma=sigma1, obs=obs)
    nll_gauss = zfit.loss.UnbinnedNLL(model=gauss_model, data=data)

    # Model 2: Mixture of two Gaussians
    mu1_mix = zfit.Parameter("mu1_model2", 0.0, -10.0, 10.0, prior=NormalPrior(mu=0.0, sigma=5.0))
    sigma1_mix = zfit.Parameter("sigma1_model2", 1.0, 0.1, 5.0, prior=HalfNormalPrior(mu=0, sigma=3.0))
    mu2_mix = zfit.Parameter("mu2_model2", 5.0, -10.0, 10.0, prior=NormalPrior(mu=5.0, sigma=5.0))
    sigma2_mix = zfit.Parameter("sigma2_model2", 1.0, 0.1, 5.0, prior=HalfNormalPrior(mu=0, sigma=3.0))
    frac = zfit.Parameter("frac", 0.5, 0.0, 1.0, prior=UniformPrior(lower=0.0, upper=1.0))

    # Add priors

    gauss1 = zfit.pdf.Gauss(mu=mu1_mix, sigma=sigma1_mix, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu=mu2_mix, sigma=sigma2_mix, obs=obs)

    mixture_model = zfit.pdf.SumPDF([gauss1, gauss2], fracs=frac)
    nll_mixture = zfit.loss.UnbinnedNLL(model=mixture_model, data=data)

    # Perform Bayesian inference for both models
    sampler_gauss = EmceeSampler()
    sampler_mixture = EmceeSampler(nwalkers=50)

    # Sample from the posterior for the Gaussian model
    print("Sampling from the Gaussian model posterior...")
    posterior_gauss = sampler_gauss.sample(nll_gauss, n_samples=3000, n_warmup=1000)

    # Sample from the posterior for the mixture model
    print("Sampling from the mixture model posterior...")
    posterior_mixture = sampler_mixture.sample(nll_mixture, n_samples=3000, n_warmup=1000)

    # Print summaries
    print("\nGaussian model results:")
    posterior_gauss.print_summary()

    print("\nMixture model results:")
    posterior_mixture.print_summary()

    # Compute Bayes factor
    log_bf = Posteriors.bayes_factor(posterior_mixture, posterior_gauss, method="stepping")

    print(f"\nLog Bayes factor (mixture / Gaussian): {log_bf:.2f}")
    if log_bf > 0:
        print(
            f"Evidence strongly favors the mixture model with exp({log_bf:.2f}) = {np.exp(log_bf):.1f} times more support"
        )
    else:
        print(f"Evidence favors the Gaussian model with exp({-log_bf:.2f}) = {np.exp(-log_bf):.1f} times more support")

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot the data and the models
    x = znp.linspace(min(data_np) - 2, max(data_np) + 2, 1000)

    # Compute posterior predictive distributions
    def predict_gauss(x_points):
        return gauss_model.pdf(x_points)

    def predict_mixture(x_points):
        return mixture_model.pdf(x_points)

    # Data histogram
    plt.subplot(2, 1, 1)
    plt.hist(data_np, bins=30, density=True, alpha=0.5, label="Data")

    # Gaussian model - posterior mean
    plt.plot(x, predict_gauss(x), "r-", label="Gaussian Model (posterior mean)")

    # Mixture model - posterior mean
    plt.plot(x, predict_mixture(x), "g--", label="Mixture Model (posterior mean)")

    # True model
    true_pdf = true_frac * 1 / (true_sigma1 * np.sqrt(2 * np.pi)) * np.exp(
        -((x - true_mu1) ** 2) / (2 * true_sigma1**2)
    ) + (1 - true_frac) * 1 / (true_sigma2 * np.sqrt(2 * np.pi)) * np.exp(-((x - true_mu2) ** 2) / (2 * true_sigma2**2))
    plt.plot(x, true_pdf, "k:", label="True Model")

    plt.xlabel(obs.label)
    plt.ylabel("Density")
    plt.title("Model Comparison")
    plt.legend()

    # Plot posterior samples for the mixture model
    plt.subplot(2, 1, 2)
    plt.scatter(
        posterior_mixture.sample(
            mu1_mix,
        ),
        posterior_mixture.sample(
            mu2_mix,
        ),
        c=posterior_mixture.sample(
            frac,
        ),
        cmap="viridis",
        alpha=0.5,
    )
    plt.colorbar(label="Fraction")
    plt.xlabel("μ₁")
    plt.ylabel("μ₂")
    plt.title("Posterior samples for mixture model")

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

    return posterior_gauss, posterior_mixture


def example_advanced_sampling():
    """Example demonstrating different sampling methods."""

    print("\n=== Comparing different sampling methods ===\n")

    # Create data
    true_mu = 2.0
    true_sigma = 1.5
    n_data = 500

    # Generate data
    data_np = np.random.normal(true_mu, true_sigma, size=n_data)
    obs = zfit.Space("x", -30, 30)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    # Define priors
    mu_prior = NormalPrior(mu=0.0, sigma=2.0)
    sigma_prior = HalfNormalPrior(mu=0, sigma=2.0)

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=mu_prior)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=sigma_prior)

    # Create model
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create negative log-likelihood loss
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Define samplers with their configurations
    verbosity = 7
    samplers = {
        # "EmceeSampler": EmceeSampler(verbosity=verbosity, nwalkers=10),
        "NUTSSampler": NUTSSampler(
            verbosity=verbosity, step_size=0.1, adapt_step_size=True, target_accept=0.8, max_tree_depth=10
        ),
        # "PTSampler": PTSampler(verbosity=verbosity, nwalkers=4, ntemps=2, adaptation_lag=1000, adaptation_time=10),
        # "SMCSampler": SMCSampler(
        #     verbosity=verbosity, n_particles=1000, n_mcmc_steps=2, ess_threshold=0.5, resampling_method="systematic"
        # ),
        # "UltraNestSampler": UltraNestSampler(
        #     verbosity=verbosity,
        #     min_num_live_points=50,
        #     cluster_num_live_points=40,
        #     dlogz=0.5,
        #     update_interval_volume_fraction=0.8,
        #     resume=True,
        # ),
    }

    # Store results
    posteriors = {}
    sampling_times = {}

    # Run sampling with each method
    for name, sampler in samplers.items():
        try:
            print(f"\nSampling with {name}...")
            start_time = time.time()

            if name == "UltraNestSampler":
                # Nested sampling doesn't use warmup
                posterior = sampler.sample(loss=nll, n_samples=500)
            else:
                posterior = sampler.sample(loss=nll, n_samples=500, n_warmup=100)

            sampling_times[name] = time.time() - start_time
            posteriors[name] = posterior

            print(
                f"{name}: μ = {posterior.mean(mu):.4f} ± {posterior.std(mu):.4f}, "
                f"σ = {posterior.mean(sigma):.4f} ± {posterior.std(sigma):.4f}"
            )
            print(f"Sampling time: {sampling_times[name]:.2f} seconds")

        except ImportError:
            print(f"Skipping {name} - not installed")
        except Exception as e:
            print(f"Error running {name}: {e!s}")

    # Plot comparison of results
    n_samplers = len(posteriors)
    if n_samplers > 0:
        plt.figure(figsize=(12, 4))

        # Plot posterior distributions for mu
        ax = plt.subplot(1, 2, 1)
        for name, posterior in posteriors.items():
            posterior.plot_posterior(mu, label=name, show_point_estimates=False, ax=ax)
        plt.axvline(true_mu, color="k", linestyle=":", label=f"True: {true_mu:.2f}")
        plt.title("Posterior for μ")
        plt.legend()

        # Plot posterior distributions for sigma
        ax = plt.subplot(1, 2, 2)
        for name, posterior in posteriors.items():
            posterior.plot_posterior(sigma, label=name, show_point_estimates=False, ax=ax)
        plt.axvline(true_sigma, color="k", linestyle=":", label=f"True: {true_sigma:.2f}")
        plt.title("Posterior for σ")
        plt.legend()

        plt.tight_layout()
        plt.savefig("sampler_comparison.png")
        plt.show()

        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 80)
        print(f"{'Sampler':<15} {'Time (s)':<10} {'μ (mean ± std)':<20} {'σ (mean ± std)':<20}")
        print("-" * 80)
        for name, posterior in posteriors.items():
            print(
                f"{name:<15} {sampling_times[name]:<10.2f} "
                f"{posterior.mean(mu):>6.4f} ± {posterior.std(mu):<6.4f}    "
                f"{posterior.mean(sigma):>6.4f} ± {posterior.std(sigma):<6.4f}"
            )

    return posteriors


def example_posterior_predictive():
    """Example of posterior predictive distributions."""

    print("\n=== Posterior predictive distributions ===\n")

    # Create data
    # np.random.seed(42)

    true_mu = 2.0
    true_sigma = 1.5
    n_data = 50  # Smaller dataset to have more uncertainty

    # Generate data
    data_np = np.random.normal(true_mu, true_sigma, size=n_data)
    obs = zfit.Space("x", -10, 10)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    # Define priors
    mu_prior = NormalPrior(mu=0.0, sigma=2.0)
    sigma_prior = HalfNormalPrior(mu=1.1, sigma=2.0)

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=mu_prior)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=sigma_prior)

    # Create model
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create negative log-likelihood loss
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Sample from posterior
    sampler = EmceeSampler()
    posterior = sampler.sample(nll, n_samples=2000, n_warmup=500)

    # Create posterior predictive distribution
    # For each posterior sample, generate a new dataset
    @tf.function(autograph=False)
    def predict_new_data(n_predict=n_data):
        return gauss.sample(n_predict).value()

    # Generate and plot the posterior predictive distribution
    predictive_samples = posterior.predictive_distribution(predict_new_data)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot original data and posterior predictive distribution
    plt.subplot(1, 2, 1)
    plt.hist(data_np, bins=20, density=True, alpha=0.7, color="blue", label="Observed data")

    # Plot some random posterior predictive datasets
    num_to_plot = 20
    indices = np.random.choice(len(predictive_samples), num_to_plot, replace=False)
    for i in indices:
        plt.hist(predictive_samples[i], bins=20, density=True, alpha=0.05, color="red")

    # Plot the average posterior predictive distribution
    all_values = np.concatenate(predictive_samples)
    plt.hist(all_values, bins=50, density=True, alpha=0.5, color="green", label="Average posterior predictive")

    # Add the true distribution
    x = np.linspace(-8, 12, 1000)
    y_true = 1 / (true_sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - true_mu) ** 2) / (2 * true_sigma**2))
    plt.plot(x, y_true, "k--", label=f"True: μ={true_mu:.2f}, σ={true_sigma:.2f}")

    plt.xlabel(obs.label)
    plt.ylabel("Density")
    plt.title("Posterior Predictive Check")
    plt.legend()

    # Plot parameter posterior with true values
    plt.subplot(1, 2, 2)
    posterior.plot_pair(mu, sigma, ax=plt.gca())
    plt.plot(true_mu, true_sigma, "r*", markersize=10, label="True values")
    plt.legend()

    plt.tight_layout()
    plt.savefig("posterior_predictive.png")
    plt.show()

    return posterior, predictive_samples


if __name__ == "__main__":
    example_gaussian_inference()
    example_model_comparison()
    # example_advanced_sampling()
    example_posterior_predictive()

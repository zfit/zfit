Bayesian Inference
=================

Introduction
-----------

zfit provides a comprehensive Bayesian inference framework that allows you to perform parameter estimation and model comparison using various MCMC (Markov Chain Monte Carlo) samplers. This functionality complements the frequentist approach of maximum likelihood estimation.

Key Components
-------------

- **Priors**: Define prior distributions for parameters
- **MCMC Samplers**: Different algorithms for sampling from posterior distributions
- **BayesianResult**: Class for analyzing and visualizing posterior distributions
- **Posterior Analysis**: Tools for analyzing and visualizing posterior distributions

BayesianResult and ZfitResult
---------------------------

The `BayesianResult` class is the main interface for working with results from Bayesian inference in zfit. It implements the `ZfitResult` interface, making it compatible with the rest of the zfit ecosystem, while providing additional functionality specific to Bayesian inference.

This means you can use a `BayesianResult` in the same way as a `FitResult` from frequentist inference:

.. code-block:: python

    # Use as a context manager to temporarily set parameters to posterior means
    with result:
        # Parameters are set to posterior means
        # Do something with the model
        pass
    # Parameters are restored to their original values

    # Update parameters to posterior means
    result.update_params()

    # Access parameter values (posterior means)
    values = result.values

    # Access parameter errors (posterior standard deviations)
    errors = result.hesse()

    # Calculate covariance matrix
    cov = result.covariance()

    # Calculate correlation matrix
    corr = result.correlation()

For backward compatibility, the `Posteriors` class is still available and is now a subclass of `BayesianResult`. However, new code should use `BayesianResult` directly.

Priors
------

zfit provides several built-in prior distributions that can be attached to parameters:

.. code-block:: python

    import zfit
    from zfit.prior import NormalPrior, HalfNormalPrior, UniformPrior

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=NormalPrior(mu=0.0, sigma=2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=HalfNormalPrior(mu=0, sigma=2.0))
    frac = zfit.Parameter("frac", 0.5, 0.0, 1.0, prior=UniformPrior(lower=0.0, upper=1.0))

MCMC Samplers
------------

zfit implements several MCMC samplers:

1. **EmceeSampler**: Ensemble sampler based on the emcee package
2. **NUTSSampler**: No-U-Turn Sampler, an efficient variant of Hamiltonian Monte Carlo
3. **PTSampler**: Parallel Tempering MCMC using ptemcee
4. **SMCSampler**: Sequential Monte Carlo sampler
5. **UltraNestSampler**: Nested sampling using UltraNest

Basic Usage
----------

Here's a simple example of Bayesian inference with zfit:

.. code-block:: python

    import zfit
    from zfit.mcmc import EmceeSampler
    from zfit.prior import NormalPrior, HalfNormalPrior

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=NormalPrior(mu=0.0, sigma=2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=HalfNormalPrior(mu=0, sigma=2.0))

    # Create a model
    obs = zfit.Space("x", -10, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create some data
    data = zfit.Data.from_numpy(obs=obs, array=[1.0, 2.0, 3.0, 1.5, 2.5])

    # Create negative log-likelihood loss
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Sample from the posterior
    sampler = EmceeSampler()
    posterior = sampler.sample(nll, n_samples=1000, n_warmup=500)

    # Analyze the results
    posterior.print_summary()

    # Access posterior statistics
    mu_mean = posterior.mean(mu)
    sigma_std = posterior.std(sigma)

Posterior Analysis
----------------

The `BayesianResult` object returned by samplers provides methods for analyzing the posterior distribution:

.. code-block:: python

    # Get posterior statistics
    mu_mean = posterior.mean(mu)
    mu_median = posterior.median(mu)
    mu_std = posterior.std(mu)

    # Get credible intervals
    lower, upper = posterior.credible_interval(mu, alpha=0.05)  # 95% credible interval

    # Get highest density interval
    hdi_lower, hdi_upper = posterior.highest_density_interval(mu, alpha=0.05)

    # Get posterior samples for a parameter
    mu_samples = posterior.sample(mu)

    # Access all posterior samples directly
    all_samples = posterior.posterior  # Same as posterior.samples

    # Plot posterior distribution
    posterior.plot_posterior(mu)

    # Plot trace (sampling history)
    posterior.plot_trace(mu)

    # Plot joint posterior (2D)
    posterior.plot_pair(mu, sigma)

    # Get a summary of the posterior
    summary = posterior.summary()
    posterior.print_summary()

Model Comparison
--------------

zfit allows for Bayesian model comparison using Bayes factors:

.. code-block:: python

    from zfit.mcmc import BayesianResult

    # Sample from two different models
    posterior1 = sampler1.sample(nll1, n_samples=1000, n_warmup=500)
    posterior2 = sampler2.sample(nll2, n_samples=1000, n_warmup=500)

    # Compute Bayes factor
    log_bf = BayesianResult.bayes_factor(posterior1, posterior2)

    if log_bf > 0:
        print(f"Evidence favors model 1 with exp({log_bf:.2f}) = {np.exp(log_bf):.1f} times more support")
    else:
        print(f"Evidence favors model 2 with exp({-log_bf:.2f}) = {np.exp(-log_bf):.1f} times more support")

Posterior Predictive Distributions
-------------------------------

You can generate posterior predictive distributions to check model fit:

.. code-block:: python

    # Generate posterior predictive samples
    predictive_samples = posterior.predictive_distribution(
        lambda: model.sample(100).value()
    )

Advanced Sampling Options
-----------------------

Each sampler has specific configuration options:

.. code-block:: python

    # NUTS sampler with custom settings
    nuts_sampler = NUTSSampler(
        step_size=0.1,
        adapt_step_size=True,
        target_accept=0.8,
        max_tree_depth=10
    )

    # Parallel Tempering sampler
    pt_sampler = PTSampler(
        nwalkers=20,
        ntemps=5,
        adaptation_lag=1000,
        adaptation_time=100
    )

    # Sequential Monte Carlo sampler
    smc_sampler = SMCSampler(
        n_particles=1000,
        n_mcmc_steps=2,
        ess_threshold=0.5,
        resampling_method="systematic"
    )

    # UltraNest sampler
    ultranest_sampler = UltraNestSampler(
        min_num_live_points=400,
        cluster_num_live_points=40,
        dlogz=0.5
    )

For more detailed examples, see the `examples/bayesian_inference.py` file in the zfit repository.

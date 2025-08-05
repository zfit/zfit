Bayesian Inference
=================

Introduction
-----------

zfit provides a Bayesian inference framework that allows you to perform parameter estimation using MCMC (Markov Chain Monte Carlo) sampling. This functionality complements the frequentist approach of maximum likelihood estimation by incorporating prior knowledge and providing full posterior distributions for parameters.

Key Components
-------------

- **Priors**: Define prior distributions for parameters
- **EmceeSampler**: MCMC sampler based on the emcee ensemble sampler
- **PosteriorSamples**: Result object for analyzing posterior distributions
- **ArviZ Integration**: Advanced diagnostics and visualization through ArviZ

Priors
------

zfit provides several built-in prior distributions that can be attached to parameters:

.. code-block:: python

    import zfit
    from zfit import prior

    # Create parameters with priors
    mu = zfit.Parameter("mu", 0.0, -5.0, 10.0, prior=prior.Normal(mu=0.0, sigma=2.0))
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 5.0, prior=prior.HalfNormal(sigma=2.0))
    frac = zfit.Parameter("frac", 0.5, 0.0, 1.0, prior=prior.Uniform(lower=0.0, upper=1.0))

Available Prior Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Normal**: Gaussian prior (automatically becomes TruncatedGauss when parameter has bounds)
- **Uniform**: Flat prior that adapts to parameter bounds
- **HalfNormal**: Positive-constrained normal for scale parameters
- **Gamma**: Flexible positive prior for rates and scales
- **Beta**: For parameters bounded between 0 and 1
- **LogNormal**: For positive parameters with log-normal uncertainty
- **Cauchy**: Heavy-tailed robust prior
- **StudentT**: Heavy-tailed alternative to normal with df parameter
- **Exponential**: For rates and waiting times
- **Poisson**: Discrete prior for count parameters
- **KDE**: Non-parametric kernel density estimate from samples

MCMC Sampling
------------

zfit currently implements the EmceeSampler, which uses the emcee ensemble sampler algorithm:

.. code-block:: python

    from zfit.mcmc import EmceeSampler

    # Create sampler with custom settings
    sampler = EmceeSampler(
        nwalkers=32,      # Number of walkers (default: 2 × n_params)
        verbosity=8       # Show progress bar
    )

    # Sample from the posterior
    posterior = sampler.sample(
        loss=nll,
        params=params,    # Optional: specify which parameters to sample
        n_samples=1000,   # Number of samples to draw
        n_warmup=500      # Number of warm-up steps
    )

Basic Usage Example
-----------------

Here's a complete example of Bayesian inference with zfit:

.. code-block:: python

    import zfit
    from zfit.mcmc import EmceeSampler
    import numpy as np

    # Create parameters with priors
    mu = zfit.Parameter("mu", 5.0, 4.5, 5.5, 
                        prior=zfit.prior.Uniform(lower=4.8, upper=5.2))
    sigma = zfit.Parameter("sigma", 0.1, 0.05, 0.3, 
                          prior=zfit.prior.HalfNormal(sigma=0.1))

    # Create a model
    obs = zfit.Space("x", -10, 10)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create some data
    data = zfit.Data.from_numpy(obs=obs, array=np.random.normal(5.0, 0.12, 1000))

    # Create negative log-likelihood loss
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

    # Sample from the posterior
    sampler = EmceeSampler(nwalkers=32, verbosity=8)
    posterior = sampler.sample(nll, n_samples=1000, n_warmup=500)

    # Display results
    print(posterior)

Posterior Analysis
----------------

The `PosteriorSamples` object returned by the sampler provides methods for analyzing the posterior distribution:

.. code-block:: python

    # Get posterior statistics
    mu_mean = posterior.mean("mu")
    mu_std = posterior.std("mu")
    
    # Get credible intervals
    lower, upper = posterior.credible_interval("mu", alpha=0.05)  # 95% CI
    
    # Get posterior samples for a parameter
    mu_samples = posterior.get_samples("mu")
    
    # Get covariance matrix
    cov = posterior.covariance()
    
    # Check convergence
    converged = posterior.converged  # Checks R̂ < 1.1 and ESS > 100
    
    # Access convergence diagnostics
    rhat = posterior.rhat  # Gelman-Rubin statistic
    ess = posterior.ess    # Effective sample size

Using Posterior Samples
~~~~~~~~~~~~~~~~~~~~~

The posterior samples integrate with zfit's parameter system:

.. code-block:: python

    # Set parameters to posterior means
    posterior.update_params()
    
    # Use as context manager to temporarily set parameters
    with posterior:
        # Parameters are set to posterior means
        model_prediction = model.sample(100)
    # Parameters are restored to original values

ArviZ Integration
---------------

For advanced diagnostics and visualization, zfit integrates with ArviZ:

.. code-block:: python

    import arviz as az
    
    # Convert to ArviZ InferenceData
    idata = posterior.to_arviz()
    
    # Use ArviZ for visualization
    az.plot_trace(idata)            # Trace plots
    az.plot_posterior(idata)        # Posterior distributions
    az.plot_pair(idata)            # Corner plots
    az.plot_autocorr(idata)        # Autocorrelation
    
    # Get summary statistics
    summary = az.summary(idata)

Hierarchical Modeling
-------------------

You can use posterior samples as priors for hierarchical modeling:

.. code-block:: python

    # Convert posterior to KDE prior
    mu_posterior_prior = posterior.as_prior("mu")
    
    # Use in a new parameter
    mu_new = zfit.Parameter("mu_new", 5.0, 4.5, 5.5, 
                           prior=mu_posterior_prior)

Warm Starting
-----------

You can continue sampling from a previous run:

.. code-block:: python

    # First run
    posterior1 = sampler.sample(nll, n_samples=500, n_warmup=200)
    
    # Continue from previous state
    posterior2 = sampler.sample(nll, n_samples=1000, init=posterior1)

Extended Models
--------------

For models with yields (extended PDFs):

.. code-block:: python

    # Create parameters with priors
    n_sig = zfit.Parameter("n_sig", 1000, 0, 5000, 
                          prior=zfit.prior.Normal(mu=1000, sigma=100))
    n_bkg = zfit.Parameter("n_bkg", 500, 0, 2000, 
                          prior=zfit.prior.Normal(mu=500, sigma=50))
    
    # Create extended PDFs
    signal = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([signal, background])
    
    # Use ExtendedUnbinnedNLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)

Convergence Diagnostics
---------------------

Monitor convergence using built-in diagnostics:

.. code-block:: python

    # Quick convergence check
    if posterior.converged:
        print("Chains have converged!")
    else:
        print(f"R-hat: {posterior.rhat}")
        print(f"ESS: {posterior.ess}")
    
    # Check for invalid samples
    if not posterior.valid:
        print("Warning: Some samples contain NaN or inf values")

Best Practices
------------

1. **Start with conservative settings**: Use more walkers and longer warm-up periods initially
2. **Check convergence**: Always verify R̂ < 1.1 and ESS > 100 for all parameters
3. **Visual inspection**: Use ArviZ trace plots to visually check for convergence
4. **Prior sensitivity**: Test how sensitive your results are to prior choices
5. **Model checking**: Use posterior predictive checks to validate your model

For a complete working example, see `examples/bayesian_inference.py` in the zfit repository.
